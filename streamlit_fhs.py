#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd


# In[2]:


st.write("# Stroke Prediction")


# In[3]:


gender = st.selectbox("Introduzca su género",["Hombre", "Mujer", "Otro"])
col1, col2, col3 = st.columns(3)

# getting user inputgender = col1.selectbox("Enter your gender",["Male", "Female"])

age = col2.number_input("Introduzca su edad")
hypertension = col3.selectbox("Tienes hipertensión?",["Sí", "No"])

heart_disease = col1.selectbox("Tienes alguna enfermedad cardiovascular?",["Sí","No"])

ever_married = col2.selectbox("Te has casado alguna vez?",["Sí", "No"])

work_type = col3.selectbox("Escoja su tipo de empleo o trabajo",["Soy menor","Empleo gubernamental", "Nunca he trabajado", "Sector privado", "Autónomo"])

Residence_type = col1.selectbox("Escoja su tipo de residencia",["Rural","Urbano"])

avg_glucose_level = col2.number_input("Introduzca la media de su nivel de glucosa en sangre")

bmi = col3.number_input("Introduzca su índice de masa corporal")
st.write("Si necesita calcular su índice de masa corporal, puede visitar la siguiente página web: [https://www.cdc.gov/healthyweight/spanish/assessing/bmi/adult_bmi/metric_bmi_calculator/bmi_calculator.html]")

smoking_status = col1.selectbox("Fuma?",["Nunca he fumado","Antes fumaba", "Soy fumador", "Prefiero no contestar"])


# In[4]:


df_pred = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])


df_pred['hypertension'] = df_pred['hypertension'].apply(lambda x: 1 if x == 'Sí' else 0)

df_pred['heart_disease'] = df_pred['heart_disease'].apply(lambda x: 1 if x == 'Sí' else 0)

df_pred['ever_married'] = df_pred['ever_married'].apply(lambda x: 1 if x == 'Sí' else 0)

df_pred['Residence_type'] = df_pred['Residence_type'].apply(lambda x: 1 if x == 'Urban' else 0)

def transform_gender(data):
    if data == 'Hombre':
        return 0
    elif data == 'Mujer':
        return 1
    else:
        return 2

def transform_work(data):
    if data == 'Soy menor':
        return 0
    elif data == 'Empleo gubernamental':
        return 1
    elif data == 'Nunca he trabajado':
        return 2
    elif data == 'Sector privado':
        return 3
    else:
        return 4

def transform_smoke(data):
    if data == 'Nunca he fumado':
        return 1
    elif data == 'Antes fumaba':
        return 0
    elif data == 'Soy fumador':
        return 2
    else:
        return 3

df_pred['gender'] = df_pred['gender'].apply(transform_gender)
df_pred['work_type'] = df_pred['work_type'].apply(transform_work)
df_pred['smoking_status'] = df_pred['smoking_status'].apply(transform_smoke)


# In[5]:


model = joblib.load('fhs_rf_model.pkl')
prediction = model.predict(df_pred)


# In[6]:


if st.button('Predict'):
    if prediction[0] == 0:
        st.write('<p class="big-font">Es poco probable que sufras un ictus/accidente cerebrovascular. Sin embargo, siempre es importante consultar a tu médico para una evaluación más completa.</p>', unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">Es probable que tengas un riesgo de ictus/accidente cerebrovascular. Es recomendable que consultes a tu médico para un diagnóstico y tratamiento adecuados.</p>', unsafe_allow_html=True)


# In[ ]:




