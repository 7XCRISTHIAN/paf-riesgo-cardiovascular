import joblib
import pandas as pd
import streamlit as st


MODELO_PATH = "modelo_logistic_regression.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNAS_PATH = "columnas_modelo.pkl"


@st.cache_resource
def cargar_recursos():
    modelo = joblib.load(MODELO_PATH)
    scaler = joblib.load(SCALER_PATH)
    columnas_modelo = joblib.load(COLUMNAS_PATH)
    return modelo, scaler, columnas_modelo


def preparar_registro(registro, columnas_modelo):
    registro_encoded = pd.get_dummies(registro, drop_first=True)
    registro_encoded = registro_encoded.reindex(columns=columnas_modelo, fill_value=0)
    return registro_encoded


st.set_page_config(
    page_title="Riesgo cardiovascular",
    page_icon="🫀",
    layout="wide",
)

modelo, scaler, columnas_modelo = cargar_recursos()

st.title("Predicción preliminar de riesgo cardiovascular")
st.caption(
    "Aplicación académica basada en Regresión Logística. "
    "No constituye diagnóstico médico."
)

st.sidebar.header("Formulario de entrada")

general_health = st.sidebar.selectbox(
    "Estado general de salud",
    ["Excellent", "Very Good", "Good", "Fair", "Poor"],
    index=2,
)
checkup = st.sidebar.selectbox(
    "Último chequeo médico",
    [
        "Within the past year",
        "Within the past 2 years",
        "Within the past 5 years",
        "5 or more years ago",
        "Never",
    ],
)
exercise = st.sidebar.selectbox("Realiza ejercicio", ["Yes", "No"])
skin_cancer = st.sidebar.selectbox("Antecedente de cáncer de piel", ["No", "Yes"])
other_cancer = st.sidebar.selectbox("Antecedente de otro cáncer", ["No", "Yes"])
depression = st.sidebar.selectbox("Antecedente de depresión", ["No", "Yes"])
diabetes = st.sidebar.selectbox(
    "Diabetes",
    [
        "No",
        "Yes",
        "No, pre-diabetes or borderline diabetes",
        "Yes, but female told only during pregnancy",
    ],
)
arthritis = st.sidebar.selectbox("Antecedente de artritis", ["No", "Yes"])
sex = st.sidebar.selectbox("Sexo", ["Female", "Male"])
age_category = st.sidebar.selectbox(
    "Categoría de edad",
    [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ],
    index=5,
)
smoking_history = st.sidebar.selectbox("Historial de tabaquismo", ["No", "Yes"])

height = st.sidebar.number_input("Estatura (cm)", min_value=90.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Peso (kg)", min_value=25.0, max_value=300.0, value=80.0)
bmi = st.sidebar.number_input("Índice de masa corporal (BMI)", min_value=12.0, max_value=100.0, value=27.5)
alcohol = st.sidebar.number_input("Consumo de alcohol", min_value=0.0, max_value=30.0, value=1.0)
fruit = st.sidebar.number_input("Consumo de frutas", min_value=0.0, max_value=120.0, value=30.0)
vegetables = st.sidebar.number_input("Consumo de vegetales verdes", min_value=0.0, max_value=128.0, value=12.0)
fried_potato = st.sidebar.number_input("Consumo de papa frita", min_value=0.0, max_value=128.0, value=4.0)
umbral = st.sidebar.slider("Umbral de decisión", 0.40, 0.80, 0.60, 0.05)

registro = pd.DataFrame(
    [
        {
            "General_Health": general_health,
            "Checkup": checkup,
            "Exercise": exercise,
            "Skin_Cancer": skin_cancer,
            "Other_Cancer": other_cancer,
            "Depression": depression,
            "Diabetes": diabetes,
            "Arthritis": arthritis,
            "Sex": sex,
            "Age_Category": age_category,
            "Height_(cm)": height,
            "Weight_(kg)": weight,
            "BMI": bmi,
            "Smoking_History": smoking_history,
            "Alcohol_Consumption": alcohol,
            "Fruit_Consumption": fruit,
            "Green_Vegetables_Consumption": vegetables,
            "FriedPotato_Consumption": fried_potato,
        }
    ]
)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Datos ingresados")
    st.dataframe(registro, use_container_width=True)
    st.info(
        "El sistema procesa variables sociodemográficas, hábitos de vida, "
        "características físicas y antecedentes de salud."
    )

with col2:
    st.subheader("Resultado")
    if st.button("Calcular riesgo", type="primary"):
        registro_modelo = preparar_registro(registro, columnas_modelo)
        registro_escalado = scaler.transform(registro_modelo)
        probabilidad = modelo.predict_proba(registro_escalado)[0, 1]
        prediccion = int(probabilidad >= umbral)

        st.metric("Probabilidad estimada", f"{probabilidad * 100:.2f}%")

        if prediccion == 1:
            st.error("Clasificación preliminar: posible riesgo cardiovascular")
            st.write(
                "El resultado sugiere revisar el caso con un profesional de salud "
                "para una evaluación preventiva."
            )
        else:
            st.success("Clasificación preliminar: riesgo bajo según el modelo")
            st.write(
                "El resultado es referencial y no reemplaza una consulta médica."
            )

        st.write(f"Umbral utilizado: **{umbral:.2f}**")

st.divider()
st.subheader("Acerca del modelo")
st.write(
    "El modelo utilizado es una Regresión Logística entrenada para estimar "
    "preliminarmente la presencia de enfermedad cardiovascular. Se utiliza como "
    "herramienta de apoyo académico y preventivo, no como diagnóstico clínico."
)
