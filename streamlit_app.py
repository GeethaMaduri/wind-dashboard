import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ----------------------------
# Helper: Get coordinates for city
# ----------------------------
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    r = requests.get(url).json()
    if "results" in r and len(r["results"]) > 0:
        return r["results"][0]["latitude"], r["results"][0]["longitude"]
    else:
        return None, None

# ----------------------------
# Helper: Fetch wind data (1 year)
# ----------------------------
def get_wind_data(lat, lon):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date=2024-01-01&end_date=2024-12-31"
        f"&hourly=windspeed_10m,winddirection_10m"
    )
    r = requests.get(url).json()
    if "hourly" not in r:
        return None

    df = pd.DataFrame(r["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    # ✅ Fix: Localize timestamps
    df["time"] = df["time"].dt.tz_localize("UTC")

    df.rename(columns={"windspeed_10m": "wind_speed", "winddirection_10m": "wind_dir"}, inplace=True)
    return df

# ----------------------------
# Helper: Estimate energy (basic power curve)
# ----------------------------
def estimate_energy(df, turbine_capacity_kw=2000):
    # Approximate cubic relation for wind turbine
    df["power_kw"] = np.where(
        (df["wind_speed"] >= 3) & (df["wind_speed"] <= 25),
        turbine_capacity_kw * (df["wind_speed"] / 12) ** 3,
        0,
    )
    return df

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🌬️ Wind Energy Feasibility Dashboard")

city = st.text_input("Enter City Name:", "Chennai")

if st.button("Get Wind Data"):
    lat, lon = get_coordinates(city)
    if lat is None:
        st.error("❌ City not found. Try another.")
    else:
        st.success(f"📍 Location: {city} ({lat}, {lon})")

        df = get_wind_data(lat, lon)
        if df is None:
            st.error("❌ Failed to fetch wind data.")
        else:
            # Estimate energy
            df = estimate_energy(df)

            # Month-wise average wind speed
            df["month"] = df["time"].dt.month
            monthly_avg = df.groupby("month")["wind_speed"].mean().reset_index()

            st.subheader("📊 Monthly Average Wind Speed")
            st.bar_chart(monthly_avg.set_index("month"))

            # Line chart of wind speed over time
            st.subheader("📈 Wind Speed Over Time")
            fig = px.line(df, x="time", y="wind_speed", title="Hourly Wind Speeds")
            st.plotly_chart(fig, use_container_width=True)

            # Histogram of wind speeds
            st.subheader("📊 Wind Speed Distribution")
            fig2 = px.histogram(df, x="wind_speed", nbins=30, title="Wind Speed Frequency")
            st.plotly_chart(fig2, use_container_width=True)

            # Potential energy output
            total_energy = df["power_kw"].sum() / 1000  # in MWh
            st.metric("Estimated Annual Energy Output", f"{total_energy:.2f} MWh")

            # ----------------------------
            # Future Forecasting (ML model)
            # ----------------------------
            st.subheader("🔮 Future Wind Speed Prediction (Simple ML)")

            df["hour"] = df["time"].dt.hour
            X = df[["hour", "month", "wind_dir"]]
            y = df["wind_speed"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            st.write(f"Model RMSE: {rmse:.2f} m/s")

            future_hours = pd.DataFrame({
                "hour": range(24),
                "month": [1]*24,
                "wind_dir": [180]*24
            })
            future_pred = model.predict(future_hours)

            fig3 = px.line(x=range(24), y=future_pred, labels={"x": "Hour", "y": "Predicted Wind Speed"},
                           title="Predicted Wind Speed for Next Day (Sample)")
            st.plotly_chart(fig3, use_container_width=True)
