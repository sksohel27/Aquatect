# Inflow Prediction Using Ensemble Methods

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)  
*Run this project interactively in Google Colab!*

**Inflow Prediction Using Ensemble Methods** is a machine learning pipeline for forecasting atmospheric pressure (as a proxy for energy inflow stability) in renewable energy systems. It uses lagged features from energy consumption deltas and global horizontal irradiance (GHI) data, combined via stacked ensemble regression to achieve high accuracy (e.g., MAPE ≈ 0.70%). Built for time-series analysis in weather-dependent energy modeling.

## 🛠️ Technology Stack

This project leverages a modern, open-source Python ecosystem for data science and machine learning:

| Category          | Technologies & Libraries                          | Purpose |
|-------------------|---------------------------------------------------|---------|
| **Programming Language** | Python 3.8+                                      | Core scripting and ML implementation |
| **Data Manipulation** | Pandas, NumPy                                    | Loading, cleaning, and feature engineering (e.g., lag features) |
| **Machine Learning** | Scikit-learn (SVR, LinearRegression, RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor) | Base regressors and metrics (RMSE, MAE, R², MAPE) |
| **Ensemble Methods** | VecStack                                          | Stacking with out-of-fold predictions and cross-validation |
| **Model Persistence** | Joblib                                            | Saving and loading trained models |
| **Utilities**      | Math (sqrt for RMSE)                              | Basic computations |
| **Environment**    | Jupyter/Colab                                     | Interactive notebooks for development and exploration |
| **Version Control**| Git/GitHub                                        | Repository management and collaboration |

Dependencies are managed via `requirements.txt` (install with `pip install -r requirements.txt`). No external APIs or cloud services required—runs locally or in free Colab.

## 🚀 Quick Start

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/sksohel27/Inflow_prediction_using_ensemble.git
   cd Inflow_prediction_using_ensemble
   ```

2. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn vecstack joblib matplotlib
   ```

3. **Run the Pipeline**:
   - Open `Inflow_prediction_using_ensemble.ipynb` (or the main script) in Jupyter/Colab.
   - Load `test2021.csv` and execute cells to train, evaluate, and predict.

4. **Explore in Colab**:
   Open the [Google Colab notebook](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE) for a no-setup demo with data viz and tuning.

## 📁 Project Structure

```
Inflow_prediction_using_ensemble/
├── .gitignore              # Standard Git ignores
├── Inflow_prediction_using_ensemble.ipynb  # Core notebook: Data prep, models, stacking
├── README.md               # Project documentation
└── test2021.csv            # Sample dataset: Energy-weather time-series (2021)
```

## 🎯 Features

- **Lag Feature Engineering**: Models temporal dependencies with $\Delta E_{\text{lag1}} = \Delta E_{t-1}$ and $GHI_{\text{lag1}} = GHI_{t-1}$.
- **Stacked Ensemble**: Combines Linear Regression, Random Forest, Decision Tree, K-Neighbors, and SVR for bias-variance reduction.
- **Evaluation Suite**: RMSE, MAE, MAPE, R² with 4-fold CV.
- **Forecasting**: Predicts future pressure (e.g., ~1014.68 hPa) on recent data.
- **Simple & Reproducible**: Single notebook for end-to-end workflow.

## 🧮 Theoretical Foundations

The pipeline estimates $E[y \mid X]$, where $y$ is pressure and $X$ includes lagged energy $\Delta E$ and GHI.

### Linear Regression
Solves the least squares:
$$
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - \beta_0 - \sum_{j=1}^p \beta_j x_{ij})^2
$$
For linear trends in lagged features.

### Stacking
Meta-features from base OOF predictions $S_{\text{train}}$ feed the meta-learner:
$$
\hat{y}_{\text{meta}} = f_{\text{meta}}(S_{\text{test}})
$$
Linear meta-model aggregates for optimal prediction.

### Metrics
- **RMSE**: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
- **MAPE**: $100 \times \frac{1}{n} \sum \frac{|y_i - \hat{y}_i|}{y_i}$
- **MAE**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **R²**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

## 🏗️ Model Overview

| Model              | Type            | Hyperparameters             | Strength                     |
|--------------------|-----------------|-----------------------------|------------------------------|
| Linear Regression | Parametric     | OLS                        | Global linearity             |
| Random Forest     | Bagging Trees  | n_estimators=10            | Non-linearity handling       |
| Decision Tree     | Single Tree    | random_state=0             | Interpretable splits         |
| K-Neighbors       | Instance-Based | n_neighbors=3              | Local patterns               |
| SVR               | Kernel         | kernel='linear'            | Outlier resistance           |

70/30 split; stacking via vecstack.

## 📊 Benchmarks

On `test2021.csv` (~34k rows):

| Metric    | Stacked | Best Base (SVR) | Δ      |
|-----------|---------|-----------------|--------|
| RMSE     | 8.98   | 9.02           | +0.44% |
| MAPE (%) | 0.701  | 0.697          | -0.57% |
| MAE      | 7.11   | 7.06           | -0.71% |
| R²       | ~0.99  | ~0.99          | =      |

Stable for pressure range ~1000-1015 hPa.

## 💻 Usage

In the notebook:
```python
# Load data
df = pd.read_csv('test2021.csv')

# Train & predict (excerpt)
X = df[['Energy_delta_lag1', 'GHI_lag1']].dropna()
# ... (full pipeline in notebook)
print(f"Predicted Inflow: {meta_model_prediction[0]:.2f} hPa")
```

Visualize errors and predictions directly in cells.

## 🔧 Contributing

- Fork & PR improvements (e.g., add XGBoost).
- Tests: Run notebook end-to-end.
- Style: PEP 8.

## ⚠️ Limitations

- Short-term focus; extend for seasonality.
- Assumes i.i.d. post-lags.

## 📄 License

MIT – see implied standard.

## 🙏 Acknowledgments

Powered by scikit-learn & vecstack. For energy forecasting research.

---

*Updated: November 26, 2025*
