import os
import joblib
import pandas as pd
import mlflow.sklearn
import boto3
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Leemos las variables tal cual se llaman en tu archivo .env
MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000') 
S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://s3:9000')
AWS_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')              
AWS_SECRET_KEY = os.getenv('MINIO_SECRET_ACCESS_KEY')       
BUCKET_NAME = os.getenv('DATA_REPO_BUCKET_NAME', 'data')


# ---------------------------------------------------------
# Agrego este if porque tuve problemas con el .env y me parece una buena práctica notificar errores
# ---------------------------------------------------------
# Verificamos qué variables faltan
missing_vars = []
if not S3_ENDPOINT_URL: missing_vars.append("MLFLOW_S3_ENDPOINT_URL")
if not AWS_ACCESS_KEY: missing_vars.append("MINIO_ACCESS_KEY")
if not AWS_SECRET_KEY: missing_vars.append("MINIO_SECRET_ACCESS_KEY")
if not BUCKET_NAME: missing_vars.append("DATA_REPO_BUCKET_NAME")

# Si falta alguna, detenemos el programa AHORA.
if missing_vars:
    error_msg = f"❌ ERROR CRÍTICO DE CONFIGURACIÓN: Faltan variables de entorno requeridas: {', '.join(missing_vars)}"
    print(error_msg)  # Esto saldrá claro en los logs de Docker
    raise ValueError(error_msg)  # Esto detiene la ejecución para no arrancar roto






app = FastAPI(title="Social Network Ads Predictor", version="1.0")
model = None
scaler = None

class PredictionRequest(BaseModel):
    Gender: str
    Age: int
    EstimatedSalary: int

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    print("⏳ Cargando artefactos...")
    # Imprimimos para verificar que leyó bien el .env (ocultando la clave por seguridad)
    print(f"🔧 Config S3: {S3_ENDPOINT_URL} | User: {AWS_ACCESS_KEY} | Bucket: {BUCKET_NAME}")
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    os.environ['AWS_ACCESS_KEY_ID'] = AWS_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = AWS_SECRET_KEY
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = S3_ENDPOINT_URL 

    # 1. Cargar Modelo
    try:
        model = mlflow.sklearn.load_model("models:/SocialNetworkAdsModel/latest")
        print("✅ Modelo cargado.")
    except Exception as e:
        print(f"❌ Error modelo: {e}")

    # 2. Cargar Scaler
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        
        # Ruta fija interna de Airflow, bucket dinámico
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key='artifacts/scaler.joblib')
        scaler_file = BytesIO(response['Body'].read())
        
        scaler = joblib.load(scaler_file)
        print("✅ Scaler cargado correctamente.")
    except Exception as e:
        print(f"❌ Error scaler: {e}")

@app.post("/predict")
@app.post("/predict")
def predict(req: PredictionRequest):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model/Scaler not loaded")

    # 1. Normalizar texto (todo a minúsculas y quitar espacios extra)
    gender_input = req.Gender.strip().lower()

    # 2. Validación estricta
    if gender_input not in ['male', 'female']:
        raise HTTPException(
            status_code=400, 
            detail=f"Valor de género inválido: '{req.Gender}'. Debe ser 'Male' o 'Female'."
        )

    # 3. Transformación (Ahora es seguro)
    # 1 para male, 0 para female
    gender_encoded = 1 if gender_input == 'male' else 0

    # Crear el DataFrame con los datos procesados
    df = pd.DataFrame([{
        "Gender": gender_encoded,
        "Age": req.Age,
        "EstimatedSalary": req.EstimatedSalary
    }])
    
    # El scaler espera recibir 'Age' y 'EstimatedSalary'.
    # Como ya creamos el DF con el gender numérico, solo escalamos las columnas numéricas.
    try:
        df[['Age', 'EstimatedSalary']] = scaler.transform(df[['Age', 'EstimatedSalary']])
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Error al escalar datos: {str(e)}")

    prediction = model.predict(df)
    return {"prediction": int(prediction[0]), "label": "COMPRA" if prediction[0]==1 else "NO COMPRA"}