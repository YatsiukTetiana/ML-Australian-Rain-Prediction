from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "aussie_rain_pipeline.joblib"


st.set_page_config(
    page_title="Australian Rain Prediction",
    page_icon="🌦️",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the complete preprocessing and prediction pipeline."""
    return joblib.load(MODEL_PATH)


st.title("🌦️ Australian Rain Prediction")

st.write(
    "Enter today's weather observations to predict "
    "whether it will rain tomorrow. Leave unknown fields empty "
    "or select 'Don't know'."
)


if not MODEL_PATH.exists():
    st.error(
        "Pipeline model not found. Make sure this file exists: "
        "`models/aussie_rain_pipeline.joblib`."
    )
    st.stop()


try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model pipeline: {exc}")
    st.stop()


# Categories used in the training dataset
UNKNOWN = "Don't know"

locations = [
    "Sydney", "Melbourne", "Brisbane", "Perth",
    "Adelaide", "Hobart", "Darwin", "Canberra",
    "Cairns", "GoldCoast", "Townsville", "AliceSprings",
    "Albany", "Albury", "WaggaWagga", "Wollongong",
    "Newcastle", "Cobar", "MountGambier", "Launceston",
    "Ballarat", "Bendigo", "Mildura", "Moree",
    "NorfolkIsland", "Nuriootpa", "PearceRAAF",
    "PerthAirport", "SydneyAirport", "MelbourneAirport",
    "Williamtown", "Woomera",
]

wind_directions = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def optional_category(label, options):
    """Display a categorical field with an unknown option."""
    return st.selectbox(
        label,
        options=[UNKNOWN, *options],
        index=0,
    )


def optional_number(label, min_value=None, max_value=None, step=1.0):
    """Display an initially empty numeric field."""
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=None,
        placeholder="Leave empty if unknown",
        step=step,
    )


with st.form("weather_input"):
    st.subheader("Location and wind")

    col1, col2 = st.columns(2)

    with col1:
        location = optional_category("Location", locations)

        wind_gust_dir = optional_category(
            "Wind gust direction", wind_directions
        )

        wind_dir_9am = optional_category(
            "Wind direction at 9 am", wind_directions
        )

    with col2:
        wind_gust_speed = optional_number(
            "Wind gust speed (km/h)", 0.0, 250.0
        )

        wind_speed_9am = optional_number(
            "Wind speed at 9 am (km/h)", 0.0, 150.0
        )

        wind_speed_3pm = optional_number(
            "Wind speed at 3 pm (km/h)", 0.0, 150.0
        )

    wind_dir_3pm = optional_category(
        "Wind direction at 3 pm", wind_directions
    )

    st.subheader("Temperature and rainfall")

    col1, col2, col3 = st.columns(3)

    with col1:
        min_temp = optional_number("Minimum temperature (°C)", step=0.5)
        max_temp = optional_number("Maximum temperature (°C)", step=0.5)

        rainfall = optional_number(
            "Rainfall (mm)", 0.0, 500.0, step=0.5
        )

        evaporation = optional_number(
            "Evaporation (mm)", 0.0, 100.0, step=0.5
        )

    with col2:
        sunshine = optional_number("Sunshine (hours)", 0.0, 24.0, step=0.5)
        temp_9am = optional_number("Temperature at 9 am (°C)", step=0.5)
        temp_3pm = optional_number("Temperature at 3 pm (°C)", step=0.5)

        rain_today = optional_category("Did it rain today?", ["No", "Yes"])

    with col3:
        humidity_9am = optional_number(
            "Humidity at 9 am (%)", 0.0, 100.0
        )

        humidity_3pm = optional_number(
            "Humidity at 3 pm (%)", 0.0, 100.0
        )

        cloud_9am = optional_number("Cloud at 9 am (oktas)", 0.0, 9.0)
        cloud_3pm = optional_number("Cloud at 3 pm (oktas)", 0.0, 9.0)

    st.subheader("Pressure")

    col1, col2 = st.columns(2)

    with col1:
        pressure_9am = optional_number(
            "Pressure at 9 am (hPa)", 850.0, 1100.0, step=0.5
        )

    with col2:
        pressure_3pm = optional_number(
            "Pressure at 3 pm (hPa)", 850.0, 1100.0, step=0.5
        )

    submitted = st.form_submit_button(
        "Predict rain tomorrow",
        type="primary",
    )


if submitted:
    input_values = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }

    # Convert "Don't know" to a missing value.
    # The categorical SimpleImputer will replace it with "missing".
    input_values = {
        column: (
            None if value == UNKNOWN else value
        )
        for column, value in input_values.items()
    }

    expected_columns = getattr(model, "feature_names_in_", None)

    if expected_columns is not None:
        missing_columns = [
            column for column in expected_columns
            if column not in input_values
        ]

        extra_columns = [
            column for column in input_values
            if column not in expected_columns
        ]

        if missing_columns or extra_columns:
            st.error(
                "The input fields do not match the model features. "
                f"Missing: {missing_columns}; extra: {extra_columns}."
            )
            st.stop()

        input_df = pd.DataFrame(
            [[input_values[column] for column in expected_columns]],
            columns=expected_columns,
        )
    else:
        input_df = pd.DataFrame([input_values])

    try:
        # The saved pipeline handles imputation, scaling and encoding.
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        positive_class = next(
            (
                candidate
                for candidate in ["Yes", 1, True, "yes", "YES"]
                if candidate in classes
            ),
            classes[-1],
        )

        positive_index = classes.index(positive_class)
        rain_probability = float(probabilities[positive_index])

        st.divider()
        st.subheader("Prediction result")

        if str(prediction).strip().lower() in {"yes", "1", "true"}:
            st.warning("Rain is predicted for tomorrow.")
        else:
            st.success("Rain is not predicted for tomorrow.")

        st.metric(
            "Probability of rain tomorrow",
            f"{rain_probability:.1%}",
        )

        st.progress(rain_probability)

        st.caption(
            "This is a machine-learning estimate, not an official "
            "weather-service forecast."
        )

    except Exception as exc:
        st.error(f"Could not make a prediction: {exc}")

        st.info(
            "Verify that the saved Pipeline was trained with the "
            "same feature names and input format used by this app."
        )