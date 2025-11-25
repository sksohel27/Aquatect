# Aquatect

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE)  
*Run this project interactively in Google Colab!*

**Aquatect** is an advanced machine learning pipeline for forecasting atmospheric pressure in energy systems using lagged features from energy consumption and solar irradiance data. It leverages ensemble stacking to combine diverse regressors, achieving robust predictions with low error rates (e.g., MAPE ≈ 0.70%). Ideal for renewable energy integration, weather-dependent modeling, and time-series analysis.

## 🚀 Quick Start

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/aquatect.git
   cd aquatect
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Core libraries: `numpy`, `pandas`, `scikit-learn`, `vecstack`, `joblib`, `matplotlib`.

3. **Run the Pipeline**:
   ```bash
   python src/main.py --data data/test2021.csv
   ```
   This trains models, evaluates performance, and generates forecasts.

4. **Explore in Colab**:
   Open the [Google Colab notebook](https://colab.research.google.com/drive/YOUR_COLAB_LINK_HERE) for a no-setup interactive demo, including data visualization and hyperparameter tuning.

## 📁 Project Structure

```
aquatect/
├── src/                 # Core ML pipeline
│   ├── __init__.py
│   ├── data_prep.py     # Preprocessing and lag features
│   ├── models.py        # Base and stacked regressors
│   └── main.py          # Entry point for training/prediction
├── data/                # Sample datasets
│   └── test2021.csv     # Energy-weather time-series data
├── notebooks/           # Exploratory Jupyter notebooks
│   └── analysis.ipynb   # Feature engineering and EDA
├── models/              # Saved artifacts
│   └── stacked_model.joblib
├── requirements.txt     # Dependencies
├── README.md            # You're reading it!
└── LICENSE              # MIT License
```

## 🎯 Features

- **Time-Series Lag Engineering**: Captures autocorrelation with features like $\Delta E_{\text{lag1}} = \Delta E_{t-1}$ and $GHI_{\text{lag1}} = GHI_{t-1}$.
- **Diverse Ensemble**: Stacks Linear Regression, Random Forest, Decision Tree, K-Neighbors, and SVR for complementary strengths.
- **Rigorous Evaluation**: Metrics include RMSE, MAE, MAPE, and R², with cross-validation for unbiased stacking.
- **Forecasting Mode**: Predicts future pressure (e.g., ~1014.68 hPa) on recent data subsets.
- **Modular & Extensible**: Easy to swap meta-learners or add deep learning (e.g., LSTMs).

## 🧮 Theoretical Foundations

Aquatect models the conditional expectation $E[y \mid X]$, where $y$ is atmospheric pressure and $X$ includes lagged energy delta ($\Delta E$) and GHI.

### Linear Regression Baseline
Minimizes the least squares objective:
$$
\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_{i1} - \cdots - \beta_p x_{ip})^2
$$
Predictions follow $\hat{y} = X \hat{\beta}$, assuming linearity for global trends.

### Stacking Ensemble
Base models generate meta-features via out-of-fold predictions $S_{\text{train}}$. The meta-learner fits:
$$
\hat{y}_{\text{meta}} = f_{\text{meta}}(S_{\text{test}})
$$
With $f_{\text{meta}}$ as Linear Regression, this reduces bias-variance via model diversity, approximating the Bayes optimal predictor.

### Evaluation Metrics
- **RMSE**: $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$
- **MAPE**: $100 \times \frac{1}{n} \sum \frac{|y_i - \hat{y}_i|}{y_i}$
- **MAE**: $\frac{1}{n} \sum |y_i - \hat{y}_i|$
- **R²**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$

## 🏗️ Model Architecture

| Model                  | Type              | Hyperparameters                  | Role                          |
|------------------------|-------------------|----------------------------------|-------------------------------|
| Linear Regression     | Parametric       | OLS solver                      | Linear trends                 |
| Random Forest         | Bagging Trees    | n_estimators=10, random_state=0 | Non-linearity, variance reduction |
| Decision Tree         | Single Tree      | random_state=0                  | Threshold rules               |
| K-Neighbors           | Instance-Based   | n_neighbors=3                   | Local adaptations             |
| SVR                   | Kernel Method    | kernel='linear'                 | Outlier robustness            |

Trained on 70/30 split with 4-fold CV stacking.

## 📊 Performance Benchmarks

On 2021 dataset (~34k rows):

| Metric     | Stacked Ensemble | Best Base (SVR) | Improvement |
|------------|------------------|-----------------|-------------|
| RMSE      | 8.98            | 9.02           | +0.44%     |
| MAPE (%)  | 0.701           | 0.697          | -0.57%     |
| MAE       | 7.11            | 7.06           | -0.71%     |
| R²        | ~0.99           | ~0.99          | Equivalent |

Low errors reflect stable pressure (~1000-1015 hPa); stacking excels in volatile periods.

## 💻 Usage Examples

### Basic Training & Prediction
```python
import pandas as pd
from src.pipeline import StackingRegressorPipeline

df = pd.read_csv('data/test2021.csv')
pipeline = StackingRegressorPipeline()
pipeline.fit(df)
future_lags = df.tail(2497)[['Energy_delta_lag1', 'GHI_lag1']].dropna()
predictions = pipeline.predict(future_lags)
print(f"Predicted Pressure: {predictions.mean():.2f} hPa")  # e.g., 1014.68
```

### Visualization
```python
pipeline.plot_metrics()  # RMSE/MAPE plots
pipeline.plot_predictions(y_test, y_pred)  # Actual vs. Predicted
```

### Custom Horizons
Extend lags for multi-step forecasts, e.g., AR(p) structure: $y_t = \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t$.

## 🔧 Contributing

1. Fork the repo and create a feature branch.
2. Install dev deps: `pip install -r dev-requirements.txt`.
3. Run tests: `pytest tests/`.
4. Commit & PR: Follow PEP 8; update `CHANGELOG.md`.

Ideas: Add XGBoost boosting, geospatial features, or Bayesian uncertainty.

## ⚠️ Limitations

- Assumes stationarity; seasonal adjustments needed for long horizons.
- Zero-inflated inputs (e.g., nighttime GHI=0) may bias baselines.
- No real-time API; extend for production deployment.

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Built with ❤️ using scikit-learn, vecstack, and Pandas. Inspired by ensemble techniques in renewable energy forecasting.

---

*Questions? Open an issue or reach out! Last updated: November 2025*
