import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Wind Energy Feasibility Dashboard", layout="wide")
st.title("🌬 Wind Energy Feasibility Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Location & Settings")
latitude = st.sidebar.number_input("Latitude", value=12.97)   # Default: Bengaluru
longitude = st.sidebar.number_input("Longitude", value=77.59)
end_date = datetime.today().date()
start_date = end_date - timedelta(days=30)
st.sidebar.write(f"Fetching wind data from {start_date} to {end_date}")

# --- Data Fetch Function ---
@st.cache_data
def get_wind_data(lat, lon, start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
        f"&hourly=windspeed_10m"
    )
    r = requests.get(url)
    data = r.json()
    if "hourly" in data:
        df = pd.DataFrame({
            "time": data["hourly"]["time"],
            "windspeed": data["hourly"]["windspeed_10m"]
        })
        df["time"] = pd.to_datetime(df["time"])
        return df
    else:
        return pd.DataFrame()

# --- Load Data ---
df = get_wind_data(latitude, longitude, start_date, end_date)

# --- Show Data ---
if not df.empty:
    st.subheader("📊 Historical Wind Speed (10m)")
    fig = px.line(df, x="time", y="windspeed", title="Hourly Wind Speed (last 30 days)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary
    avg_speed = df["windspeed"].mean()
    st.metric("Average Wind Speed (m/s)", f"{avg_speed:.2f}")

    # Histogram
    st.subheader("Wind Speed Distribution")
    hist = px.histogram(df, x="windspeed", nbins=20, title="Wind Speed Frequency")
    st.plotly_chart(hist, use_container_width=True)

    # --- Energy Estimation ---
    def estimate_power(wind_speed, rated_power=2000, cut_in=3, rated=12):
        if wind_speed < cut_in:
            return 0
        elif wind_speed <= rated:
            return rated_power * ((wind_speed - cut_in) / (rated - cut_in))**3
        else:
            return rated_power

    df["power_output"] = df["windspeed"].apply(estimate_power)
    total_energy = df["power_output"].sum() / 1000  # kWh approx (hourly)

    st.subheader("⚡ Energy Estimation")
    st.metric("Estimated Energy (kWh)", f"{total_energy:,.0f}")

    # --- Forecast ---
    st.subheader("📈 Wind Speed Forecast (Next 24 Hours)")
    model = ARIMA(df["windspeed"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(24)
    st.line_chart(forecast)

else:
    st.error("No wind data available for this location and time range.")