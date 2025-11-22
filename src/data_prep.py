import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import s3fs
import os

# --- CONFIGURACIÓN ---
# Puerto 9090 (Host) -> 9000 (Contenedor)
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT_URL', 'http://127.0.0.1:9090')
MINIO_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
MINIO_SECRET = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
BUCKET_NAME = 'data'
FILE_NAME = 'Social_Network_Ads.csv'

# Configuración de S3FS 
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': MINIO_ENDPOINT},
    key=MINIO_KEY,
    secret=MINIO_SECRET,
    use_ssl=False
)

# Opciones para Pandas (reusando la configuración de fs)
# Pandas acepta 'storage_options' para no tener que pasarle el objeto fs entero
S3_OPTIONS = {
    'client_kwargs': {'endpoint_url': MINIO_ENDPOINT},
    'key': MINIO_KEY,
    'secret': MINIO_SECRET,
    'use_ssl': False
}

def load_data():
    path = f's3://{BUCKET_NAME}/{FILE_NAME}'
    print(f"-> Leyendo: {path}")
    return pd.read_csv(path, storage_options=S3_OPTIONS)

def save_data(df, filename):
    path = f's3://{BUCKET_NAME}/processed/{filename}'
    print(f"-> Guardando datos: {path}")
    df.to_csv(path, index=False, storage_options=S3_OPTIONS)

def save_artifact(obj, filename):
    """
    Guarda un objeto binario (scaler) usando s3fs como puente para joblib.
    """
    path = f'{BUCKET_NAME}/artifacts/{filename}' 
    print(f"-> Guardando artefacto: s3://{path}")
    
    # 1. Abrimos el archivo remoto con s3fs
    # 2. Le pasamos ese archivo abierto a joblib
    with fs.open(path, 'wb') as f:
        joblib.dump(obj, f)

def prepare_data():
    # 1. Cargar
    df = load_data()
    
    # 2. Preprocesamiento
    df = df.drop('User ID', axis=1, errors='ignore')
    df['Gender'] = df['Gender'].apply(lambda x: True if x == 'Male' else False)
    y = df['Purchased'].astype(int)
    X = df[['Gender', 'Age', 'EstimatedSalary']]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Escalado (SOLO variables continuas)
    COLUMNS_TO_SCALE = ['Age', 'EstimatedSalary']
    scaler = StandardScaler()
    
    X_train.loc[:, COLUMNS_TO_SCALE] = scaler.fit_transform(X_train[COLUMNS_TO_SCALE])
    X_test.loc[:, COLUMNS_TO_SCALE] = scaler.transform(X_test[COLUMNS_TO_SCALE])
    
    # 5. Guardar Scaler 
    save_artifact(scaler, 'scaler.joblib')
    
    # 6. Guardar CSVs
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    save_data(train_df, 'train_scaled.csv')
    save_data(test_df, 'test_scaled.csv')
    
    print("\n✅ Pipeline finalizado correctamente.")

if __name__ == "__main__":
    prepare_data()
