import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import plotly.express as px

# Load the trained model
model = pickle.load(open('house_price_prediction_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# Title of the app
st.title(" California Housing Price Predictator")
st.image('House_bg.jpg')

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'latitude' not in st.session_state:
    st.session_state.latitude = None
if 'longitude' not in st.session_state:
    st.session_state.longitude = None
if 'address' not in st.session_state:
    st.session_state.address = None

# Input fields for user
st.header("Enter the attributes of the housing:")
latitude = st.number_input("Latitude", format= "%0.2f", value=37.88)
longitude = st.number_input("Longitude", format="%0.2f", value=-122.23)
housing_median_age = st.slider("Housing Median Age", 1, 100, 30)
total_rooms = st.number_input("Total Rooms within a block", min_value=1, step=1, value= 880)
total_bedrooms = st.number_input("Total Bedrooms within a block", min_value=1, step=1, value=129)
population = st.number_input("Population within a block", min_value=1, step=1, value=322)
households = st.number_input("Households for a block", min_value=1, step=1, value=126)
median_income = st.number_input("Median Income within a block", format="%0.4f", value=8.3252)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

# Predict button
if st.button("Predict Price"):
    ocean_categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    ocean_encoded = [1 if ocean_proximity == category else 0 for category in ocean_categories]

    input_data = np.array([[latitude, longitude, housing_median_age, total_rooms, 
                                total_bedrooms, population, households, median_income] + ocean_encoded])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    st.session_state['prediction'] = prediction[0]
    st.session_state['latitude'] = latitude
    st.session_state['longitude'] = longitude

    geolocator = Nominatim(user_agent="house_price_app")
    location = geolocator.reverse((latitude, longitude))
    st.session_state.address = location.address if location else "Address not found"


# Show results if stored
if st.session_state.prediction is not None:
    st.subheader("📈 Predicted Price")
    st.success(f"💲 {st.session_state.prediction:,.2f}")

    st.markdown(
        "**📝 Note:** This prediction is based on average data per block. "
        "So, the price is an estimate of house values in that local area, not just one exact house."
    )

    st.subheader("📍 Location")
    st.info(f"{st.session_state.address}")

 
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=12)
    folium.Marker(
        [st.session_state.latitude, st.session_state.longitude], 
        popup=f"Predicted Price: ${st.session_state.prediction:,.2f}", 
        icon=folium.Icon(icon="home", prefix='fa', color='blue')
    ).add_to(m)
    st.subheader("🗺️ House Location on Map")
    st_folium(m, width=700, height=500)

    # Simulate 100 house prices around the predicted average to reflect block variation
    nearby_prices = np.random.normal(
        loc=st.session_state.prediction,
        scale=st.session_state.prediction * 0.15,  # Adjust spread based on prediction
        size=100
    )

    # Store the nearby prices in session state so they remain consistent for this prediction
    st.session_state['nearby_prices'] = nearby_prices

    # Show visualizations only on button click
    if st.button("📊 Show Visualizations") and 'nearby_prices' in st.session_state:
        nearby_prices = st.session_state['nearby_prices']
        prediction = st.session_state.prediction

        df = pd.DataFrame({
            "House Index": range(1, 101),
            "Price": nearby_prices,
            "Income Level": np.random.randint(1, 6, size=100)  # For optional animation later
        })

        col1, col2 = st.columns(2)

        with col1:
            scatter_fig = px.scatter(
            df,
            x="House Index",
            y="Price",
            title="House Prices Around Predicted Average",
            color="Price",
            color_continuous_scale="Blues",
            template="plotly_white"
        )

            scatter_fig.add_hline(
            y=prediction,
            line_dash="dash",
            line_color="red",
            annotation_text="Predicted Avg",
            annotation_position="top left"
        )

            st.plotly_chart(scatter_fig, use_container_width=True)

        with col2:
            box_fig = px.box(
            df,
            y="Price",
            title="Price Distribution in Block",
            points="all",  # Show all points for better understanding
            template="plotly_white"
        )

            st.plotly_chart(box_fig, use_container_width=True)
