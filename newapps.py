#Deskrispsi : Dashboard ini digunakan untuk mendeteksi apakah seseorang memiliki Diabetes atau tidak.

#import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

#Membuat Judul dan Sub-Judul
st.write("""
# Dashboard Deteksi Diabetes
Deteksi apakah seseorang memiliki Diabetes 
""")

#Load data nya
df = pd.read_csv("diabetes/train.csv")

#Buat Subheader
st.subheader("Data Information")

#Load dataframe nya
st.dataframe(df)

#load statistik data nya
st.write(df.describe())

#Show chart Data nya
st.subheader("Plot Information untuk 50 pasien pertama")
chart=st.bar_chart(df.head(50))

#Split the data into independet 'X' dan dependent "Y" variable
X = df.iloc[:,1:9].values
Y = df.iloc[:,-1].values

#Split the data set into 80% training dan 20%testing
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Input dari User
def get_user_input():
    number_pregnancies = st.sidebar.slider("no_times_pregnant",0,17,3)
    glucose_concentration = st.sidebar.slider("glucose_concentration",0,199,117)
    blood_pressure = st.sidebar.slider("blood_pressure",0,122,72)
    skin_fold_thickness = st.sidebar.slider("skin_fold_thickness",0,99,23)
    serum_insulin = st.sidebar.slider("serum_insulin",0.0,846.0,30.5)
    bmi = st.sidebar.slider("bmi",0.0,67.1,30.0)
    diabetes_pedigree = st.sidebar.slider("diabetes pedigree",0.078,2.42,0.3725)
    age = st.sidebar.slider("age",21,81,29)
    
    #Store a dictionary to variable
    user_data = {"no_times_pregnant" : number_pregnancies,
                  "glucose_concentration" : glucose_concentration,
                  "blood_pressure" : blood_pressure,
                  "skin_fold_thickness" : skin_fold_thickness,
                  "serum_insulin" : serum_insulin,
                  "bmi" : bmi,
                   "diabetes pedigree" : diabetes_pedigree,
                   "age" : age  
                    }
    
    #Transform data jadi data frame
    features = pd.DataFrame(user_data,index=[0])
    return features

#Simpan user input jadi variabel
user_input = get_user_input()\

#Set a subheader and display user input
st.subheader("User Input:")
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train,Y_train)

#Show Model Metric
st.subheader('Model Test Accuracy Score: ')
st.write( str(accuracy_score(Y_test,RandomForestClassifier.predict(X_test))*100)+"%")

#Simpan Prediksi Model di dalam variable
prediction = RandomForestClassifier.predict(user_input)

#Set a Subheader and display the classification
st.subheader("Classification:")
st.write(prediction)
st.subheader("0 = non-diabetic, 1 = potentially diabetic")
