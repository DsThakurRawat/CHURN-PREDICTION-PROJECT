import pickle
import os

for fname in ["onehot_encoder_geo.pkl", "label_encoder_gender.pkl", "scaler.pkl"]:
    print(fname, "exists:", os.path.exists(fname))

try:
    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)
except Exception as e:
    print("Error loading onehot_encoder_geo:", e)
try:
    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)
except Exception as e:
    print("Error loading label_encoder_gender:", e)
try:
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except Exception as e:
    print("Error loading scaler:", e)


## integrating ann with streamlit app
import pickle

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
import streamlit as st
import numpy as np

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelBinarizer,OneHotEncoder
import pandas as pd
import pickle
#now loading the trained model
model = tf.keras.models.load_model("model.h5")
#loadin all encoder all of these things
#streamlit app

st.title("customer churn prediction")
#user input
geography = st.selectbox("Geography",onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure",0,10)
num_of_products = st.slider("Number of Products",1,4)
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Member",[0,1])

#prepare the input data
input_data = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Geography" : [geography],
    "Gender" : [gender],
    "Age" : [age],

    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary]
})
# Label encode the Gender column before one-hot encoding Geography
input_data['Gender'] = label_encoder_gender.transform(input_data["Gender"])

#one hot encode "Geography"
geo_encoded = onehot_encoder_geo.transform(input_data[["Geography"]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = onehot_encoder_geo.get_feature_names_out(["Geography"]))

#combine one-hot encoded columns with inut data
input_data = pd.concat([input_data.drop("Geography",axis =1),geo_encoded_df],axis =1)


#scale the input data
input_data_scaled = scaler.transform(input_data)

#make prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")

