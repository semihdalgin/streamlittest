import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

st.title("House Price Prediction App")
st.write("""
This app uses a Scikit-Learn machine learning model (KNN Regressor) 
to estimate the sale price of a house based on the features you input.
""")

#st.divider()


@st.cache_resource
def load_model():
    return pickle.load(open('models/trained_pipe_knn.sav', 'rb'))


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found! Please make sure the 'models/trained_pipe_knn.sav' file is in the correct directory.")
    st.stop()

st.subheader("Enter House Features")

col1, col2 = st.columns(2)

with col1:
    LotArea = st.number_input("Lot Area (sq ft)", min_value=0, value=9000, step=100)
    TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, value=1000, step=50)

with col2:
    BedroomAbvGr = st.number_input("Number of Bedrooms", min_value=0, value=3, step=1)
    GarageCars = st.number_input("Garage Capacity (Cars)", min_value=0, value=2, step=1)

st.write("")


if st.button("Predict Price 🚀", use_container_width=True):

    new_house = pd.DataFrame({
        'LotArea': [LotArea],
        'TotalBsmtSF': [TotalBsmtSF],
        'BedroomAbvGr': [BedroomAbvGr],
        'GarageCars': [GarageCars]
    })


    prediction = model.predict(new_house)[0]


    st.success(f"Estimated House Sale Price: **${prediction:,.2f}**")
    st.balloons()
