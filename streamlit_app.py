# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide", page_title="Wind Energy Dashboard 🌬️")

# ----------------------------
# Turbine Power Curves
# ----------------------------
TURBINES = {
    "Generic 2 MW": {"rated_power":2000, "cut_in":3, "cut_out":25, "rated_wind":12},
    "Siemens SWT-2.3MW": {"rated_power":2300, "cut_in":3, "cut_out":25, "rated_wind":12},
    "GE 1.5MW": {"rated_power":1500, "cut_in":3, "cut_out":25, "rated_wind":12},
}

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def get_coordinates(city_name):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    try:
        r = requests.get(url, timeout=15).json()
        if "results" in r and len(r["results"]) > 0:
            return (r["results"][0]["latitude"],
                    r["results"][0]["longitude"],
                    r["results"][0].get("country",""))
        return None, None, None
    except:
        return None, None, None

@st.cache_data(show_spinner=False)
def fetch_wind_data(lat, lon, start_year, end_year):
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=windspeed_10m,winddirection_10m"
        f"&timezone=UTC"
    )
    try:
        r = requests.get(url, timeout=60).json()
        if "hourly" not in r:
            return pd.DataFrame()
        df = pd.DataFrame(r["hourly"])
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        df.rename(columns={"windspeed_10m":"wind_speed","winddirection_10m":"wind_dir"}, inplace=True)
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["hour"] = df["time"].dt.hour
        return df
    except:
        return pd.DataFrame()

def estimate_energy_power_curve(df, turbine_name="Generic 2 MW"):
    """
    Estimates energy using turbine-specific power curve.
    Also calculates capacity factor.
    """
    t = TURBINES[turbine_name]
    rated = t["rated_power"]
    cut_in = t["cut_in"]
    cut_out = t["cut_out"]
    rated_wind = t["rated_wind"]
    
    df = df.copy()
    df["power_kw"] = np.where(
        (df["wind_speed"] < cut_in) | (df["wind_speed"] > cut_out),
        0,
        np.where(df["wind_speed"] <= rated_wind,
                 rated * (df["wind_speed"]/rated_wind)**3,
                 rated)
    )
    df["capacity_factor"] = df["power_kw"]/rated
    return df

def train_rf_model(df):
    X = df[["hour","month","wind_dir"]]
    y = df["wind_speed"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=-1)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    return model, rmse

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌬 Wind Energy Dashboard")

# Step 1: Get location
city = st.text_input("Enter City Name:", "Chennai")

if city:
    lat, lon, country = get_coordinates(city)
    if lat is None:
        st.error("❌ City not found. Try 'City, CountryCode' format (e.g., 'Chennai, IN').")
    else:
        st.success(f"📍 Location: {city}, {country} ({lat:.2f}, {lon:.2f})")

        # Turbine selection
        turbine_selected = st.selectbox("Select Turbine:", list(TURBINES.keys()))

        # Step 2: Fetch 2020-2024 data
        st.header("📊 Historical Wind Data: 2020–2024")
        df_full = fetch_wind_data(lat, lon, 2020, 2024)
        if df_full.empty:
            st.error("❌ No wind data found for 2020–2024 for this location.")
        else:
            df_full = estimate_energy_power_curve(df_full, turbine_selected)

            # Monthly averages
            monthly_avg = df_full.groupby("month")["wind_speed"].mean().reset_index()
            st.subheader("Average Monthly Wind Speed (2020–2024)")
            st.bar_chart(monthly_avg.set_index("month"))

            # Yearly summary
            yearly_avg = df_full.groupby("year")["wind_speed"].mean().reset_index()
            st.subheader("Average Yearly Wind Speed (2020–2024)")
            st.line_chart(yearly_avg.set_index("year"))

            # Hourly time series
            st.subheader("Hourly Wind Speeds (2020–2024)")
            st.plotly_chart(px.line(df_full, x="time", y="wind_speed"), use_container_width=True)

            # Wind speed distribution
            st.subheader("Wind Speed Distribution (2020–2024)")
            st.plotly_chart(px.histogram(df_full, x="wind_speed", nbins=30), use_container_width=True)

            # Energy & capacity factor
            total_energy_mwh = df_full["power_kw"].sum()/1000
            avg_capacity_factor = df_full["capacity_factor"].mean()
            st.metric("Estimated Total Energy Output (2020–2024)", f"{total_energy_mwh:.2f} MWh")
            st.metric("Average Capacity Factor", f"{avg_capacity_factor:.2%}")

            # Monthly energy output
            monthly_energy = df_full.groupby("month")["power_kw"].sum()/1000
            st.subheader("Monthly Energy Output (MWh)")
            st.bar_chart(monthly_energy)

            # Capacity factor over time
            st.subheader("Capacity Factor Over Time")
            st.line_chart(df_full[["time","capacity_factor"]].set_index("time"))

        # Step 3: Optional filters
        st.header("🔍 Explore Specific Year/Month")
        years = list(range(2020, 2025))
        selected_year = st.selectbox("Select Year:", years)
        months = ["All"] + [str(i) for i in range(1,13)]
        selected_month = st.selectbox("Select Month (Optional):", months)

        df_filtered = df_full[df_full["year"]==selected_year]
        if selected_month != "All":
            df_filtered = df_filtered[df_filtered["month"]==int(selected_month)]
            st.info(f"Showing data for {selected_year}, Month: {selected_month} ({len(df_filtered)} hourly entries)")

        if not df_filtered.empty:
            st.subheader(f"Wind Data for {selected_year}" + (f", Month {selected_month}" if selected_month!="All" else ""))
            st.dataframe(df_filtered.head(30))

            st.plotly_chart(px.line(df_filtered, x="time", y="wind_speed", title="Hourly Wind Speeds"), use_container_width=True)
            st.plotly_chart(px.histogram(df_filtered, x="wind_speed", nbins=30, title="Wind Speed Distribution"), use_container_width=True)

            total_energy_filtered = df_filtered["power_kw"].sum()/1000
            avg_cf_filtered = df_filtered["capacity_factor"].mean()
            st.metric("Estimated Energy Output", f"{total_energy_filtered:.2f} MWh")
            st.metric("Average Capacity Factor", f"{avg_cf_filtered:.2%}")

        # Step 4: 2025 Forecast
        st.header("🔮 Wind Prediction: 2025 (Next Year Forecast)")
        df_2025 = fetch_wind_data(lat, lon, 2025, 2025)
        if df_2025.empty:
            st.warning("⚠️ 2025 forecast data not available. Using simple model based on 2020–2024")
            df_2025 = df_full.copy()

        df_2025 = estimate_energy_power_curve(df_2025, turbine_selected)
        model, rmse = train_rf_model(df_2025)
        st.write(f"Model RMSE: {rmse:.2f} m/s")

        future_hours = pd.date_range("2025-01-01", periods=24, freq="H", tz="UTC")
        X_future = pd.DataFrame({
            "hour": future_hours.hour,
            "month": future_hours.month,
            "wind_dir": [int(df_2025['wind_dir'].median())]*24
        })
        preds = model.predict(X_future)
        future_df = pd.DataFrame({"time":future_hours,"pred_wind_speed":preds})
        st.plotly_chart(px.line(future_df, x="time", y="pred_wind_speed",
                                title="Predicted Wind Speed for 2025 (Sample)"), use_container_width=True)

        st.caption("Note: 2025 prediction uses a simple Random Forest model based on historical wind patterns. For professional studies, advanced models are recommended.")

