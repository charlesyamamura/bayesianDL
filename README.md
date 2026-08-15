# bayesianDL
Bayesian Deep Learning for New Product Demand Forecasting

This repository implements a Bayesian Deep Learning (BDL) pipeline in JAX and Equinox to forecast automotive market share while providing calibrated epistemic uncertainty bounds. Designed specifically for tabular market data subject to temporal drift, the model leverages a Last-Layer Laplace Approximation (LL-Laplace) to combine deep feature extraction with robust Bayesian inference.

Key Features & Methodology
Architecture & Framework: Built with Equinox and JAX, utilizing explicit functional partitioning and JIT compilation. The network uses a 2-hidden-layer LeakyReLU MLP (width=32) with weight decay regularization (optax.adamw).
Temporal Drift Mitigation: Numeric features are scaled relative to annual market averages (X 
i,t
​	
 / 
X
ˉ
  
t
​	
 ) prior to min-max normalization. This keeps multi-year features on a shared relative manifold across changing market conditions.
Last-Layer Laplace (LL-Laplace): Instead of sampling across dense weight spaces—which causes predictive drift in deep non-linear networks—the model freezes MAP hidden representations (Φ) and analytically evaluates the closed-form posterior over the output layer:
Σ 
last
​	
 =( 
σ 
obs
2
​	
 
Φ 
T
 Φ
​	
 +λI) 
−1
 
Uncertainty Quantification: Computes closed-form epistemic variance (σ 
epistemic
2
​	
 =ϕ(x) 
T
 Σ 
last
​	
 ϕ(x)) and exports 89% High-Density Intervals (HDI) formatted for ArviZ.
Model Interpretability:
Gradient Sensitivity Analysis: Quantifies global feature importance via expected absolute gradients (E[ 

​	
  
∂x 
j
​	
 
∂ 
y
^
​	
 
​	
  

​	
 ]).
Partial Dependence Plots (PDPs): Maps marginal feature effects across the market spectrum with shaded 90% Bayesian confidence bands.
Performance Metrics
Evaluated on an out-of-time test split (Training: ≤2018, Test: 2019):

Dataset	R 
2
  Score	RMSE	MAE
Train (≤2018)	0.906	0.239	0.158
Test (2019)	0.798	0.373	0.280
Tech Stack
Core ML: jax, equinox, optax
Bayesian Diagnostics & Stats: arviz, scikit-learn
Data & Visualization: pandas, numpy, matplotlib
Outputs Generated
performance_and_uncertainty.png: Actual vs. predicted market share and out-of-sample HDI uncertainty error bars.
partial_dependence_plots.png: Marginal effect curves for top product drivers bounded by epistemic uncertainty.
