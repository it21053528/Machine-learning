import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Diabetes Prediction App

#### This app predicts wheather the patient is Diabetic or not

""")

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    Pregnancies = st.sidebar.number_input('Pregnancies', 0 ,60, 1)
    Glucose = st.sidebar.number_input('Glucose', 0, 200, 126)
    BloodPressure = st.sidebar.number_input('BloodPressure', 0.0, 150.0, 60.0)
    SkinThickness = st.sidebar.number_input('SkinThickness', 0.1, 100.0, 29.0)
    Insulin = st.sidebar.number_input('Insulin', 0.0, 1000.0, 125.0)
    BMI = st.sidebar.number_input('BMI', 0.0, 70.0, 30.1)
    DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', 0.0, 3.0, 0.349)
    Age = st.sidebar.number_input('Age',0, 120, 47)
    data = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'Age':Age
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reads in saved classification model
load_clf = pickle.load(open('diabetes.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('User Input')
st.write(input_df)

st.subheader('Prediction')
diabetic_type = np.array(['Non-Diabetic','Diabetic'])
st.write(diabetic_type[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
