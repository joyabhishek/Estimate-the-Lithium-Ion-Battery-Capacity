import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from math import sqrt
import time

st.title('Estimate the Lithium-Ion Battery Capacity based on the current readings')

def RMSE(y, y_pred):
  sum=0
  for ac, pred in zip(y, y_pred):
    sum+= (ac-pred)**2
  return sqrt(sum/len(y))


DATA_URL = ('https://raw.githubusercontent.com/ignavinuales/Battery_RUL_Prediction/main/Datasets/HNEI_Processed/Final%20Database.csv')
scaler = StandardScaler()

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data = data.drop(['Unnamed: 0'], axis=1)
    return data

@st.cache
def preprocess_data(data_copy, discharge_Time_median, decrement_median, time_median, time_Constant_Current_median, scaler_model):

    #Fix Skew in data
    data_copy.loc[(data_copy["Discharge Time (s)"] < (818.837200)) | (data_copy["Discharge Time (s)"] > (2590.020001)),"Discharge Time (s)"] = discharge_Time_median
    data_copy.loc[(data_copy["Decrement 3.6-3.4V (s)"] < (191.178043)) | (data_copy["Decrement 3.6-3.4V (s)"] > (1367.584331)),"Decrement 3.6-3.4V (s)"] = decrement_median
    data_copy.loc[(data_copy["Time at 4.15V (s)"] < (944.768750)) | (data_copy["Time at 4.15V (s)"] > (5209.249481)),"Time at 4.15V (s)"] = time_median
    data_copy.loc[(data_copy["Time constant current (s)"] < (1484.380000)) | (data_copy["Time constant current (s)"] > (6196.357400)),"Time constant current (s)"] = time_Constant_Current_median
    
    #Scale the data
    scaled_data = scaler_model.transform(data_copy[['Cycle_Index', 'Discharge Time (s)','Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)','Time constant current (s)']])
    data_copy.loc[:,['Cycle_Index', 'Discharge Time (s)','Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)','Time constant current (s)']] = scaled_data

    return data_copy

@st.cache
def train_model(preprocessed_data):
    X = preprocessed_data[['Cycle_Index','Discharge Time (s)','Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)','Time constant current (s)']]
    y = preprocessed_data[['RUL']]

    lin_reg = LinearRegression()
    reg = lin_reg.fit(X, y)

    return reg


data_load_state = st.text('Loading data...')
#Load all rows of data into the dataframe.
data = load_data()
data_load_state.text('Loading data...done!!')

#Create a copy of the raw data
data_copy = data.copy()

discharge_Time_median  = data_copy["Discharge Time (s)"].median()
decrement_median = data_copy["Decrement 3.6-3.4V (s)"].median()
time_median = data_copy["Time at 4.15V (s)"].median()
time_Constant_Current_median = data_copy["Time constant current (s)"].median()

scaler_model = scaler.fit(data_copy[['Cycle_Index', 'Discharge Time (s)','Decrement 3.6-3.4V (s)', 'Time at 4.15V (s)','Time constant current (s)']])

#Lets pre process the data
preprocessed_data = preprocess_data(data_copy, discharge_Time_median, decrement_median, time_median, time_Constant_Current_median, scaler_model)

#Model train
lin_reg_model = train_model(preprocessed_data)

#Get data from user
cycle_Index = st.text_input('Cycle Index', '1')
discharge_Time  = st.text_input('Discharge Time (s)', '2595.30')
decrement = st.text_input('Decrement 3.6-3.4V (s)', '1151.488500')
time = st.text_input('Time at 4.15V (s)', '5460.001')
time_Constant_Current = st.text_input('Time constant current (s)', '6755.01')

predict_rul_button = st.button("Predit RUL")
if predict_rul_button:
    user_data = {'Cycle_Index': [float(cycle_Index)], 'Discharge Time (s)': [float(discharge_Time)], 'Decrement 3.6-3.4V (s)': [float(decrement)], 'Time at 4.15V (s)': [float(time)], 'Time constant current (s)': [float(time_Constant_Current)]}
    user_data =pd.DataFrame.from_dict(user_data)
    preprocessed_user_data = preprocess_data(user_data, discharge_Time_median, decrement_median, time_median, time_Constant_Current_median, scaler_model)
    user_data_pred  = lin_reg_model.predict(preprocessed_user_data)
    st.write(f"Predicted RUL: {user_data_pred[0,0]}")

    life = int(user_data_pred/1200 * 100)
    st.write(f"Life left: {life}%")
    progress = st.progress(0)
    for i in range(life):
        t.sleep(0.01)
        progress.progress(i)
