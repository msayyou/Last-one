import pandas as pd
import streamlit as st
import joblib

st.title('Scoring Credit App')

st.sidebar.header("the  parameters of the client")


def client_parameters_enter():
    CODE_GENDER = st.sidebar.selectbox('CODE_GENDER', ('M', 'F'))
    EXT_SOURCE_3 = st.sidebar.slider('EXT_SOURCE_3', 0.00, 1.00, 0.01)
    EMERGENCYSTATE_MODE = st.sidebar.selectbox('EMERGENCYSTATE_MODE', ('No', 'Yes'))
    OCCUPATION_TYPE = st.sidebar.selectbox('OCCUPATION_TYPE', ('Laborers', 'Drivers', 'Sales staff', 'Cleaning',
                                                               'staff',
                                                               'Managers', 'Security staff', 'Accountants',
                                                               'Core staff',
                                                               'Realty agents', 'Medicine staff',
                                                               'High skill tech staff', 'Cooking staff',
                                                               'Secretaries', 'Low-skill Laborers', 'IT staff',
                                                               'Private service staff',
                                                               'HR staff', 'Waiters/barmen staff'))
    NAME_INCOME_TYPE = st.sidebar.selectbox('NAME_INCOME_TYPE', ('Working', 'Commercial associate', 'State servant',
                                                                 'Businessman', 'Student'))
    NAME_EDUCATION_TYPE = st.sidebar.selectbox('NAME_EDUCATION_TYPE', ('Secondary / secondary special',
                                                                       'Higher education',
                                                                       'Incomplete higher', 'Lower secondary',
                                                                       'Academic degree'))
    WALLSMATERIAL_MODE = st.sidebar.selectbox('WALLSMATERIAL_MODE', ('Stone, brick', 'Panel', 'Others',
                                                                     'Monolithic', 'Mixed', 'Wooden',
                                                                     'Block'))
    REGION_RATING_CLIENT = st.sidebar.slider('REGION_RATING_CLIENT', 1, 3, 1)

    data = {'CODE_GENDER': CODE_GENDER, 'EXT_SOURCE_3': EXT_SOURCE_3, 'EMERGENCYSTATE_MODE': EMERGENCYSTATE_MODE,
            'OCCUPATION_TYPE': OCCUPATION_TYPE, 'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
            'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE, 'WALLSMATERIAL_MODE': WALLSMATERIAL_MODE,
            'REGION_RATING_CLIENT': REGION_RATING_CLIENT}

    client_parameters = pd.DataFrame(data, index=[0])
    return client_parameters


input_df = client_parameters_enter()

st.header('Deployment')

df = pd.read_csv('data_model.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

donnee_entree = pd.concat([input_df, df], axis=0)
st.write(donnee_entree)

var_cat = ['CODE_GENDER', 'EMERGENCYSTATE_MODE', 'OCCUPATION_TYPE',
           'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'WALLSMATERIAL_MODE']

for col in var_cat:
    dummy = pd.get_dummies(donnee_entree[col], drop_first=True)
    donnee_entree = pd.concat([dummy, donnee_entree], axis=1)
    del donnee_entree[col]

# prendre uniquement la premiere ligne
donnee_entree = donnee_entree.fillna(donnee_entree.mode())
donnee_entree = donnee_entree[:1]

tableau_prevision = donnee_entree.drop(['TARGET'], axis=1)
st.subheader('Les nouveaux parametres')
st.write(tableau_prevision)

X = donnee_entree.drop(['TARGET'], axis=1)
y = donnee_entree['TARGET']

# importer le modèle
load_model = joblib.load('predict_loan.pkl')

# appliquer le modèle sur le profil d'entrée
prevision = load_model.predict(tableau_prevision)
prediction_proba = load_model.predict_proba(tableau_prevision)

st.subheader('Résultat de la prévision')
st.write(y[prevision])

st.subheader('prediction probability')
st.write(prediction_proba)
