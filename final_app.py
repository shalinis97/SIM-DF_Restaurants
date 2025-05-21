import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import sqlite3
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta
import joblib
from dotenv import load_dotenv
import os
import json
import requests
# Load environment variables from .env file
load_dotenv()

# Retrieve variables from the environment
latitude = float(os.getenv("LATITUDE", "0.0"))
longitude = float(os.getenv("LONGITUDE", "0.0"))
timezone = os.getenv("TIMEZONE", "UTC")
weather_file = os.getenv("WEATHER_FILE", "weather.json")
base_url = os.getenv("BASE_URL","")
# -----------------------------------------------------------------------------
# Data Loading and Preparation (from SQLite)
# -----------------------------------------------------------------------------
# Read Excel files and load into SQLite (only if db doesn't exist)
if not os.path.exists("data.db"):
    sales_df = pd.read_excel("Sales Dataset OG.xlsx")
    prod_df = pd.read_excel("Production.xlsx")
    recipe_df = pd.read_excel("Food reciepe.xlsx")
    conn = sqlite3.connect("data.db")
    sales_df.to_sql("sales", conn, if_exists="replace", index=False)
    prod_df.to_sql("production", conn, if_exists="replace", index=False)
    recipe_df.to_sql("recipe", conn, if_exists="replace", index=False)
    conn.close()

# Query combined data from SQLite
conn = sqlite3.connect("data.db")
query = '''
SELECT s.[System Date] as Date, s.[Food Name] as Food_Name, s.Quantity as Quantity_Sold, 
       p.[Production Food Name] as Production, r.Ingredients
FROM sales s
LEFT JOIN production p ON s.[Food Name] = p.[Production Food Name]
LEFT JOIN recipe r ON s.[Food Name] = r.[Food Name]
'''
combined_df = pd.read_sql_query(query, conn, parse_dates=["Date"])
conn.close()

# Clean and sort combined_df
combined_df = combined_df.dropna(subset=["Date", "Food_Name", "Quantity_Sold"])
combined_df["Date"] = pd.to_datetime(combined_df["Date"], errors="coerce")
combined_df = combined_df.sort_values(["Food_Name", "Date"]).reset_index(drop=True)

st.set_page_config(
    page_title="Restaurant Dish Demand Forecasting",
    layout="centered",
)

# Animated Heading using st.markdown and HTML/CSS
st.markdown(
    '''
    <h1 style="text-align:center; font-size:2.8rem; font-family: 'Segoe UI', sans-serif;">
        <span class="fade-in">🍽️ Restaurant Dish Demand Forecasting</span>
    </h1>
    <style>
    .fade-in {
        animation: fadeIn 2s ease-in;
        display: inline-block;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .sidebar-anim {
        animation: slideInLeft 1.5s cubic-bezier(.68,-0.55,.27,1.55);
        display: block;
    }
    @keyframes slideInLeft {
        0% { opacity: 0; transform: translateX(-40px); }
        100% { opacity: 1; transform: translateX(0); }
    }
    </style>
    ''',
    unsafe_allow_html=True
)

FORECAST_HORIZON_DAYS = 7  # 7-Day Forecast

@st.cache_data(show_spinner=False)
def unique_dishes(df: pd.DataFrame) -> list[str]:
    """Get list of unique dish names."""
    return sorted(df["Food_Name"].unique().tolist())

@st.cache_data(show_spinner=False)
def load_recipe_df():
    return pd.read_excel("Food reciepe.xlsx")

recipe_df = load_recipe_df()

# ---- Weather Data Integration Functions ----

def load_file_weath():
    if os.path.exists(weather_file):
        with open(weather_file, "r") as fw:      
            return fw.readlines()
    return None

st.features_rf = [
            "sales_diff_7", "sales_diff_14", "sales_diff",
            "lag_1_day_sales", "rolling_14_day_avg", "rolling_7_day_avg",
            "lag_14_day_sales", "lag_7_day_sales", "temperature"
        ]


def weatherdata(date_str,lines_fw):
    try:
        for line in lines_fw:
            try:
                data = json.loads(line)
                if data["daily"]["time"][0] == date_str:
                    # If date already exists, use it
                    #print(f"Using cached data for {date_str}")
                    day_data = data["daily"]
                    return {
                        "temperature": (day_data["temperature_2m_max"][0] + day_data["temperature_2m_min"][0]) / 2,
                        "weather_main": "Rain" if day_data["precipitation_sum"][0] > 0 else "Clear"
                    },lines_fw
            except Exception as e:
                continue  # Skip bad lines
    except Exception as e:
            print(f"Error retrieving weather data from file")
    # If not found in file, call API
    try:
        url = (
            f"{base_url}"
            f"latitude={latitude}&longitude={longitude}"
            f"&start_date={date_str}&end_date={date_str}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
            f"&timezone={timezone}"
        )

        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            # Append to file
            with open(weather_file, "a") as f:
                f.write(json.dumps(data) + "\n")

            lines_fw = load_file_weath()

            day_data = data["daily"]
            return {
                "temperature": (day_data["temperature_2m_max"][0] + day_data["temperature_2m_min"][0]) / 2,
                "weather_main": "Rain" if day_data["precipitation_sum"][0] > 0 else "Clear",   
            },lines_fw

        else:
            print(f"Open-Meteo API error on {date_str}: Status {response.status_code}")
            return None

    except Exception as e:
        print(f"Error on {date_str}: {e}")
        return None




st.sidebar.markdown('<div class="sidebar-anim"><h2>🔍 Select Dish</h2></div>', unsafe_allow_html=True)
dish_names = unique_dishes(combined_df)
selected_dish = st.sidebar.selectbox("Dish Name", dish_names, index=0, key="dish_select")


lines_fw = load_file_weath()
st.sidebar.markdown("---")


# Ensure R² and MAE values are calculated and accessible globally
st.r2 = 0.0
st.mae = 0.0


# Main Display and Model Analysis toggle
view_option = st.sidebar.radio("View Options", ["📈 Restaurant View", "📊 Model Analysis"], key = "view_option")


@st.cache_resource(show_spinner="🔄 Training Random Forest model...")
def train_model(df_feat, features_rf):
    split = int(len(df_feat) * 0.8)
    X_train, y_train = df_feat[features_rf][:split], df_feat["Quantity_Sold"][:split]
    X_test, y_test = df_feat[features_rf][split:], df_feat["Quantity_Sold"][split:]

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    model = random_search.best_estimator_
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, y_test, y_pred, r2, mae, split




if selected_dish:
    dish_df = combined_df.query("Food_Name == @selected_dish")
    if dish_df.empty:
        st.warning("No data for selected dish.")
        st.stop()
    # --- Feature Engineering ---
    st.df_feat = dish_df[["Date", "Quantity_Sold"]].copy()
    st.df_feat = st.df_feat.set_index("Date").resample("D").sum().fillna(0).reset_index()

    st.df_feat["sales_diff"] = st.df_feat["Quantity_Sold"].diff()
    st.df_feat["sales_diff_7"] = st.df_feat["Quantity_Sold"].diff(7)
    st.df_feat["sales_diff_14"] = st.df_feat["Quantity_Sold"].diff(14)
    st.df_feat["lag_1_day_sales"] = st.df_feat["Quantity_Sold"].shift(1)
    st.df_feat["lag_7_day_sales"] = st.df_feat["Quantity_Sold"].shift(7)
    st.df_feat["lag_14_day_sales"] = st.df_feat["Quantity_Sold"].shift(14)
    st.df_feat["rolling_7_day_avg"] = st.df_feat["Quantity_Sold"].shift(1).rolling(7).mean()
    st.df_feat["rolling_14_day_avg"] = st.df_feat["Quantity_Sold"].shift(1).rolling(14).mean()
    # --- Weather integration for this dish ---:
    st.df_feat["weather_date"] = st.df_feat["Date"].dt.strftime("%Y-%m-%d")
    # Weather data integration
    st.df_feat["temperature"] = 0.0

    print(f"Loading weather data for {selected_dish}...")
    for i, row in st.df_feat.iterrows():
        date_str = row["weather_date"]
        weather, lines_fw = weatherdata(date_str, lines_fw)
        if weather:
            st.df_feat.at[i, "temperature"] = weather["temperature"]


    print(f"Weather data loaded for {selected_dish}.")

    st.df_feat = st.df_feat.dropna().reset_index(drop=True)

    
    # split = int(len(st.df_feat) * 0.8)
    # X_train, y_train = st.df_feat[st.features_rf][:split], st.df_feat["Quantity_Sold"][:split]
    # X_test, y_test = st.df_feat[st.features_rf][split:], st.df_feat["Quantity_Sold"][split:]
    # param_dist = {
    #     "n_estimators": [100, 200, 300, 500],
    #     "max_depth": [5, 10, 15, None],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "max_features": ["sqrt", "log2", None],
    #     "bootstrap": [True, False]
    # }
    # rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    # random_search = RandomizedSearchCV(
    #     estimator=rf,
    #     param_distributions=param_dist,
    #     n_iter=30,
    #     cv=3,
    #     scoring='r2',
    #     verbose=0,
    #     random_state=42,
    #     n_jobs=-1
    # )
    # random_search.fit(X_train, y_train)

    # # Calculate R² and MAE after model training
    # st.model = random_search.best_estimator_
    # y_pred = st.model.predict(X_test)
    # st.mae = mean_absolute_error(y_test, y_pred)
    # st.r2 = r2_score(y_test, y_pred)


    st.model, st.y_test, st.y_pred, st.r2, st.mae, split = train_model(st.df_feat, st.features_rf)


    
  


if view_option == "📈 Restaurant View":
    st.subheader(f"📈 Forecast for: {selected_dish}")
    # Plot Actual vs Predicted on test set
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(st.df_feat["Date"][split:], st.y_test, label="Actual", linewidth=2)
    ax.plot(st.df_feat["Date"][split:], st.y_pred, label="Predicted", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantity Sold")
    ax.set_title("Actual vs Predicted (Test Set)")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    st.pyplot(fig)
    forecast_rows = []
    df_extended = st.df_feat.copy()
    last_date = df_extended["Date"].max()

    for i in range(7):
        forecast_date = last_date + timedelta(days=i + 1)
        feature_row = {}
        weather_date = forecast_date.strftime("%Y-%m-%d")
        temperature = 0.0
        weather, lines_fw = weatherdata(weather_date, lines_fw)
        if weather:
            temperature = weather["temperature"]
        feature_row["temperature"] = temperature
        recent_sales = df_extended["Quantity_Sold"]
        feature_row["lag_1_day_sales"] = recent_sales.iloc[-1] if len(recent_sales) >= 1 else 0
        feature_row["lag_7_day_sales"] = recent_sales.iloc[-7] if len(recent_sales) >= 7 else 0
        feature_row["lag_14_day_sales"] = recent_sales.iloc[-14] if len(recent_sales) >= 14 else 0
        feature_row["rolling_7_day_avg"] = recent_sales[-7:].mean() if len(recent_sales) >= 7 else 0
        feature_row["rolling_14_day_avg"] = recent_sales[-14:].mean() if len(recent_sales) >= 14 else 0
        feature_row["sales_diff"] = recent_sales.diff().iloc[-1] if len(recent_sales) >= 2 else 0
        feature_row["sales_diff_7"] = recent_sales.diff(7).iloc[-1] if len(recent_sales) >= 7 else 0
        feature_row["sales_diff_14"] = recent_sales.diff(14).iloc[-1] if len(recent_sales) >= 14 else 0
        feature_row["Date"] = forecast_date
        X_pred = pd.DataFrame([feature_row])[st.features_rf]
        pred_quantity = st.model.predict(X_pred)[0].round()
        feature_row["forecast_quantity"] = int(pred_quantity)
        forecast_rows.append(feature_row)
        df_extended = pd.concat([
            df_extended,
            pd.DataFrame({"Date": [forecast_date], "Quantity_Sold": [pred_quantity]})
        ], ignore_index=True)
    forecast_df = pd.DataFrame(forecast_rows)
    st.subheader("🔮 7‑Day Demand Forecast (RF)")
    st.dataframe(forecast_df[["Date", "forecast_quantity"]], hide_index=True)
    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    recent_history = st.df_feat.tail(20)[["Date", "Quantity_Sold"]]
    ax2.plot(recent_history["Date"], recent_history["Quantity_Sold"], label="Historical", linewidth=2)
    ax2.plot(forecast_df["Date"], forecast_df["forecast_quantity"], label="Forecast", marker="o")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Quantity Sold")
    ax2.set_title("Last 20 Days + 7‑Day Forecast (RF)")
    ax2.legend()
    ax2.grid(True, linestyle=":", linewidth=0.5)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    # Ingredient forecast (unchanged)
    ingredients = recipe_df[recipe_df["Food Name"] == selected_dish][["Ingredients", "I_Quantity"]]
            
    if not ingredients.empty:
        # For each forecasted day, multiply predicted quantity by I_Quantity for each ingredient
        ingredient_forecast = []
        for _, row in forecast_df.iterrows():
            for _, ing in ingredients.iterrows():
                ingredient_forecast.append({
                    "Date": row["Date"].date(),
                    "Ingredient": ing["Ingredients"],
                    "Total_Required": row["forecast_quantity"] * ing["I_Quantity"]
                })
        ingredient_forecast_df = pd.DataFrame(ingredient_forecast)
        st.subheader("🧮 7‑Day Ingredient Requirement Forecast")
        st.dataframe(ingredient_forecast_df, hide_index=True)
    else:
        st.info("No ingredient mapping found for this dish in Food reciepe.xlsx.")
    
    st.subheader("🧮 Total Ingredient Required for 7 Days")
    st.dataframe(
        ingredient_forecast_df.groupby("Ingredient")["Total_Required"].sum().reset_index(),
        hide_index=True
        )
elif view_option == "📊 Model Analysis":
    st.subheader(f"📊 Model Analysis for: {selected_dish}")
    # Display R² Score and MAE
    st.metric(label="R² Score", value=f"{st.r2:.3f}")
    st.metric(label="MAE", value=f"{st.mae:.2f}")

    st.subheader("ℹ️ Feature Importance (Top 10)")

    if hasattr(st.model, 'feature_importances_'):
        importance = (
            pd.DataFrame({
                "feature": st.features_rf,
                "importance": st.model.feature_importances_
            })
            .sort_values("importance", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )
        st.dataframe(importance, use_container_width=True)
    else:
        st.info("Feature importance not available for this model.")

st.sidebar.markdown("---")
st.sidebar.write("© 2025 – Demand Forecasting App")