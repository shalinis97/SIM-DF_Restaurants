# 🍽️ SMART INVENTORY MANAGEMENT FOR RESTAURANTS AND CAFETERIA

A robust Streamlit web app for forecasting restaurant dish demand and ingredient requirements using machine learning (RandomForestRegressor/XGBoost) and historical sales data. The app supports per-dish models, weather-aware features, and interactive visualizations.

---

## Features

- **Per-dish demand forecasting** with Random Forest or XGBoost models
- **Automatic feature engineering** (lags, rolling means, calendar, weather)
- **Weather integration** (temperature as a feature for select dishes)
- **Ingredient requirement forecasting** based on recipes and predicted sales
- **Interactive Streamlit UI** with animated headings and sidebar
- **Visualizations**: Actual vs. Predicted, 7-day forecast, ingredient breakdown
- **SQLite backend** for efficient data management
- **Automatic model retraining** if no model exists for a dish

---

## Project Structure

```
├── app.py                  # Main Streamlit app
├── data.db                 # SQLite database (auto-generated)
├── Sales Dataset OG.xlsx   # Sales data (Excel)
├── Production.xlsx         # Production data (Excel)
├── Food reciepe.xlsx       # Recipe/ingredient mapping (Excel)
├── pkl files/              # Folder for per-dish trained model .pkl files
│   └── <Dish>_best_rf_model.pkl
└── ...
```

---

## Getting Started

### 1. Install Requirements

```bash
pip install streamlit pandas numpy scikit-learn matplotlib xgboost openpyxl
```

### 2. Prepare Data

- Place your sales, production, and recipe Excel files in the project root.
- The app will auto-create `data.db` on first run.

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Using the App

- Select a dish from the sidebar.
- View demand forecasts, ingredient requirements, and feature importances.
- If a model is missing or outdated, delete its `.pkl` file to retrain.

---

## Customization

- **Weather Integration**: Edit the `weatherdata` and `load_file_weath` functions in `app.py` to use real weather APIs or files.
- **Feature Engineering**: Adjust the `engineer_features` function for new features.
- **Modeling**: Switch between RandomForestRegressor and XGBoost as needed.

---

## License

MIT License

---

## Credits

- Developed by Shalini ,Himanshu and Jaival
- Powered by Streamlit, scikit-learn, XGBoost, and pandas
