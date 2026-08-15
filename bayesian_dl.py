import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.flatten_util
import equinox as eqx
import optax
import arviz as az
from jaxtyping import Float, Array, PRNGKeyArray, PyTree
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------------------------------------------------------
# 1. Load Data & Apply Relative Feature Scaling
# -----------------------------------------------------------------------------
df = pd.read_excel("data1319.xlsx")

train_df = df[df['year'] <= 2018].copy()
test_df  = df[df['year'] == 2019].copy()

target_col = "share"
exclude_cols = [target_col, "mo", "year", "model"]

# Feature scaling relative to annual averages to mitigate temporal market drift
numeric_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype != 'object']
for col in numeric_cols:
    train_df[col] = train_df[col] / train_df.groupby('year')[col].transform('mean')
    test_df[col]  = test_df[col] / test_df.groupby('year')[col].transform('mean')

X_train_raw = train_df.drop(columns=exclude_cols)
X_test_raw  = test_df.drop(columns=exclude_cols)
y_train_raw = train_df[target_col].values
y_test_raw  = test_df[target_col].values

# Categorical Label Encoding
for col in X_train_raw.select_dtypes(include=["object", "category"]).columns:
    le = LabelEncoder()
    X_train_raw[col] = le.fit_transform(X_train_raw[col])
    X_test_raw[col]  = le.transform(X_test_raw[col])

in_scaler = MinMaxScaler()
out_scaler = MinMaxScaler()

X_tr_jax = jnp.array(in_scaler.fit_transform(X_train_raw), dtype=jnp.float32)
y_tr_jax = jnp.array(out_scaler.fit_transform(y_train_raw.reshape(-1, 1)), dtype=jnp.float32)
X_te_jax = jnp.array(in_scaler.transform(X_test_raw), dtype=jnp.float32)
y_te_jax = jnp.array(out_scaler.transform(y_test_raw.reshape(-1, 1)), dtype=jnp.float32)

# -----------------------------------------------------------------------------
# 2. Network Architecture (Explicit Feature Extractor + Last Layer)
# -----------------------------------------------------------------------------
class RobustMarketMLP(eqx.Module):
    feature_extractor: list
    last_layer: eqx.nn.Linear

    def __init__(self, in_size: int, out_size: int, width_size: int, depth: int, key: PRNGKeyArray):
        keys = jr.split(key, depth + 1)
        layers = [eqx.nn.Linear(in_size, width_size, key=keys[0])]
        for i in range(depth - 1):
            layers.append(eqx.nn.Linear(width_size, width_size, key=keys[i + 1]))
        self.feature_extractor = layers
        self.last_layer = eqx.nn.Linear(width_size, out_size, key=keys[-1])

    def extract_features(self, x: Array) -> Array:
        for layer in self.feature_extractor:
            x = jax.nn.leaky_relu(layer(x))
        return x

    def __call__(self, x: Array) -> Array:
        phi = self.extract_features(x)
        return self.last_layer(phi)

key = jr.PRNGKey(42)
key_model, key_train = jr.split(key)

model = RobustMarketMLP(in_size=X_tr_jax.shape[1], out_size=1, width_size=32, depth=2, key=key_model)

# -----------------------------------------------------------------------------
# 3. MAP Training with Optax (Weight Decay Regularization)
# -----------------------------------------------------------------------------
optimizer = optax.adamw(learning_rate=0.002, weight_decay=1e-2)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

@eqx.filter_jit
def train_step(m: RobustMarketMLP, opt_st: optax.OptState, xb: Array, yb: Array):
    def loss_fn(mod):
        preds = jax.vmap(mod)(xb)
        return jnp.mean((preds - yb) ** 2)
    loss_val, grads = jax.value_and_grad(loss_fn)(m)
    updates, opt_st = optimizer.update(grads, opt_st, m)
    return eqx.apply_updates(m, updates), opt_st, loss_val

print("Training MAP estimate via Optax...")
epochs = 500
batch_size = 32
n_samples = X_tr_jax.shape[0]

for epoch in range(epochs):
    key_train, subkey = jr.split(key_train)
    perm = jr.permutation(subkey, n_samples)
    X_sh, y_sh = X_tr_jax[perm], y_tr_jax[perm]
    for i in range(0, n_samples, batch_size):
        model, opt_state, _ = train_step(model, opt_state, X_sh[i:i+batch_size], y_sh[i:i+batch_size])

# -----------------------------------------------------------------------------
# 4. Last-Layer Laplace (LL-Laplace) Posterior Computation
# -----------------------------------------------------------------------------
print("Fitting Last-Layer Laplace posterior...")

# Extracted representations (Phi) for training and test sets
Phi_train = jax.vmap(model.extract_features)(X_tr_jax)  # (n_train, 32)
Phi_test  = jax.vmap(model.extract_features)(X_te_jax)   # (n_test, 32)

map_preds_tr = jax.vmap(model)(X_tr_jax)
map_preds_te = jax.vmap(model)(X_te_jax)

# Estimate empirical observation noise variance
sigma_sq = float(jnp.var(y_tr_jax - map_preds_tr)) + 1e-5
prior_prec = 5.0

# Precision and covariance for last-layer weights: K = (Phi^T Phi)/sigma^2 + lambda * I
feature_dim = Phi_train.shape[1]
K_last = (Phi_train.T @ Phi_train) / sigma_sq + prior_prec * jnp.eye(feature_dim)
cov_last = jnp.linalg.inv(K_last)

# Closed-form predictive epistemic variance: phi_i^T Cov phi_i
var_epistemic_tr = jnp.sum((Phi_train @ cov_last) * Phi_train, axis=1)
var_epistemic_te = jnp.sum((Phi_test @ cov_last) * Phi_test, axis=1)

std_tr_scaled = np.sqrt(np.maximum(0.0, np.array(var_epistemic_tr)))
std_te_scaled = np.sqrt(np.maximum(0.0, np.array(var_epistemic_te)))

# -----------------------------------------------------------------------------
# 5. Bayesian Predictive Inference & Inverse Scaling
# -----------------------------------------------------------------------------
n_post_samples = 500
np.random.seed(42)

samples_tr_s = np.array(map_preds_tr).flatten() + np.random.normal(0, 1, (n_post_samples, len(X_tr_jax))) * std_tr_scaled
samples_te_s = np.array(map_preds_te).flatten() + np.random.normal(0, 1, (n_post_samples, len(X_te_jax))) * std_te_scaled

pred_tr = np.stack([
    out_scaler.inverse_transform(np.clip(samples_tr_s[i], 0.0, 1.0).reshape(-1, 1)).flatten()
    for i in range(n_post_samples)
])
pred_te = np.stack([
    out_scaler.inverse_transform(np.clip(samples_te_s[i], 0.0, 1.0).reshape(-1, 1)).flatten()
    for i in range(n_post_samples)
])

y_train_pred_mean = np.mean(pred_tr, axis=0)
y_test_pred_mean  = np.mean(pred_te, axis=0)

# -----------------------------------------------------------------------------
# 6. Performance Metrics
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

train_metrics = compute_metrics(y_train_raw, y_train_pred_mean)
test_metrics  = compute_metrics(y_test_raw, y_test_pred_mean)

results_df = pd.DataFrame([
    {"Model": "LL-Laplace Bayesian DL", "Dataset": "Train (<=2018)", **train_metrics},
    {"Model": "LL-Laplace Bayesian DL", "Dataset": "Test (2019)",    **test_metrics}
])

print("\n" + "="*60)
print("PERFORMANCE RESULTS (TRAIN vs TEST)")
print("="*60)
print(results_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 7. ArviZ Data Structuring & Plotting
# -----------------------------------------------------------------------------
idata_test = az.from_dict(
    posterior_predictive={"share": np.expand_dims(pred_te, axis=0)},
    observed_data={"share": y_test_raw}
)

hdi_test = az.hdi(idata_test.posterior_predictive["share"])["share"].values

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Train Set Predictions
axes[0].scatter(y_train_raw, y_train_pred_mean, alpha=0.5, color="#1f77b4", edgecolors="none")
min_tr, max_tr = y_train_raw.min(), y_train_raw.max()
axes[0].plot([min_tr, max_tr], [min_tr, max_tr], "k--", lw=1.5, label="1:1 Line")
axes[0].set_title(f"Train Set: True vs Predicted Mean\n($R^2 = {train_metrics['R2']:.3f}$, RMSE = {train_metrics['RMSE']:.3f})")
axes[0].set_xlabel("True Market Share (%)")
axes[0].set_ylabel("Predicted Market Share (%)")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.4)

# Plot 2: Test Set Predictions with ArviZ Epistemic Uncertainty
sort_idx = np.argsort(y_test_raw)
y_test_sorted = y_test_raw[sort_idx]
pred_mean_sorted = y_test_pred_mean[sort_idx]
hdi_sorted = hdi_test[sort_idx]

yerr_lower = np.maximum(0.0, pred_mean_sorted - hdi_sorted[:, 0])
yerr_upper = np.maximum(0.0, hdi_sorted[:, 1] - pred_mean_sorted)

axes[1].errorbar(
    range(len(y_test_raw)), 
    pred_mean_sorted, 
    yerr=[yerr_lower, yerr_upper],
    fmt="o", color="#2ca02c", ecolor="#98df8a", elinewidth=1.2, capsize=2, alpha=0.8,
    label="Posterior Mean & 89% HDI (ArviZ)"
)
axes[1].plot(range(len(y_test_raw)), y_test_sorted, "r.", label="True Market Share", alpha=0.9)
axes[1].set_title(f"Test Set (2019): Epistemic Uncertainty\n($R^2 = {test_metrics['R2']:.3f}$, RMSE = {test_metrics['RMSE']:.3f})")
axes[1].set_xlabel("Test Samples (Sorted by True Share)")
axes[1].set_ylabel("Market Share (%)")
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# 8. Feature Importance (Gradient Sensitivity Analysis)
# -----------------------------------------------------------------------------
feature_names = X_train_raw.columns.tolist()

# Compute mean absolute gradient per feature across the training set
grad_fn = jax.vmap(jax.grad(lambda x: model(x)[0]))
abs_grads = np.abs(np.array(grad_fn(X_tr_jax)))
mean_importance = np.mean(abs_grads, axis=0)

# Normalize attributions to 100%
importance_pct = (mean_importance / np.sum(mean_importance)) * 100

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance (%)": importance_pct,
    "Mean Abs Gradient": mean_importance
}).sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)

print("\n" + "="*60)
print("TOP FEATURE IMPORTANCE RANKING")
print("="*60)
print(importance_df.head(10).to_string(index=False))

# -----------------------------------------------------------------------------
# 9. Partial Dependence Plots (PDCs) with Bayesian Bounds
# -----------------------------------------------------------------------------
top_n = 4
top_feature_indices = [feature_names.index(col) for col in importance_df["Feature"].head(top_n)]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

grid_points = np.linspace(0, 1, 50)
scale_factor = float(out_scaler.data_max_[0] - out_scaler.data_min_[0])

for idx, feat_idx in enumerate(top_feature_indices):
    feat_name = feature_names[feat_idx]
    pdp_means, pdp_stds = [], []
    
    for val in grid_points:
        # Vary target feature across grid while fixing remaining sample values
        X_temp = np.array(X_tr_jax).copy()
        X_temp[:, feat_idx] = val
        X_temp_jax = jnp.array(X_temp)
        
        # Representations & Predictions
        Phi_temp = jax.vmap(model.extract_features)(X_temp_jax)
        preds_temp = jax.vmap(model)(X_temp_jax).flatten()
        
        # Epistemic variance along the grid slice
        var_ep = jnp.sum((Phi_temp @ cov_last) * Phi_temp, axis=1)
        std_ep = np.sqrt(np.maximum(0.0, np.array(var_ep)))
        
        pdp_means.append(float(jnp.mean(preds_temp)))
        pdp_stds.append(float(np.mean(std_ep)))
        
    pdp_means_unscaled = out_scaler.inverse_transform(np.array(pdp_means).reshape(-1, 1)).flatten()
    pdp_stds_unscaled = np.array(pdp_stds) * scale_factor
    
    ax = axes[idx]
    ax.plot(grid_points, pdp_means_unscaled, color="#1f77b4", lw=2.5, label="Expected Market Share")
    ax.fill_between(
        grid_points, 
        pdp_means_unscaled - 1.645 * pdp_stds_unscaled, 
        pdp_means_unscaled + 1.645 * pdp_stds_unscaled, 
        color="#1f77b4", alpha=0.2, label="90% Bayesian Epistemic Band"
    )
    
    ax.set_title(f"Key Driver: {feat_name}", fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Relative {feat_name} (0 = Market Min, 1 = Market Max)")
    ax.set_ylabel("Predicted Market Share (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

plt.tight_layout()
plt.show()