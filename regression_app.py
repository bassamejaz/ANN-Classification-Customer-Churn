import pandas as pd, numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import streamlit as st


#loading the trained model
model = tf.keras.models.load_model("regression_model.h5")

# Loading the encoders and scalar
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl","rb") as file:
    one_hot_encoder = pickle.load(file)

with open("regression_scaler.pkl","rb") as file:
    scalar = pickle.load(file)

# Streamlit app

st.title("Customer Salary Prediction")

#User Input
geography = st.selectbox("Geography",one_hot_encoder.categories_[0])
gender = st.selectbox("Gender",label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of products",1,4)
has_cr_card = st.selectbox("Has credit card",[0,1])
is_active_number = st.selectbox("Is Active Member",[0,1])
exited = st.selectbox("Is Exited",[0,1])

input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    "Tenure":[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_number],
    "Exited":[exited]
})

values = one_hot_encoder.transform([[geography]]).toarray()
columns = one_hot_encoder.get_feature_names_out(["Geography"])

geo_df = pd.DataFrame(values,columns=columns)
new_df = pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)

scaled_df = scalar.transform(new_df)

prediction = model.predict(scaled_df)

pred_salary = prediction[0][0]

st.write("Salary of customer",round(pred_salary,4))

