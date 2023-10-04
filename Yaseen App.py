# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 14:58:12 2023
@author: Gujrat laptops
"""

import numpy as np
import pickle as pk
import streamlit as st

loaded_model = pk.load(open('C:/Users/DELL/Downloads/trained_heartdisease_model.sav', 'rb'))

def Prediction_Function(input_data):
    input_data_as_numpy = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)

    if(prediction[0] == 0):
        return 'The Person have not a Disease'
    else:
        return 'The Person have a Disease'

def main():
    
    st.title('Disease Prediction Web App')
    
    Age = st.number_input('Enter Your Age')
    Sex=st.number_input('Sex')
    ChestPainType =st.number_input('ChestPain Type')
    RestingBPBP=st.number_input('Resting BP')
    Cholesterol=st.number_input('Cholesterol Value')
    FastingBS =st.number_input('Fasting BS')
    RestingECG=st.number_input('Resting ECG')
    MaxHR=st.number_input('Max HR')
    ExerciseAngina=st.number_input('Exercise Angina')
    Oldpeak=st.number_input('Old Peak')
    ST_Slope=st.number_input('ST slope')
    
    prediction = ''
    
    if st.button('Check Result'):
        prediction= Prediction_Function([Age,Sex,ChestPainType,RestingBPBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
        
    st.success(prediction)
    
if __name__ == '__main__':
    main()
 