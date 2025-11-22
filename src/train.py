import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
import os

# --- CONFIGURACIÓN ---
# 1. Definimos las variables con os.getenv para que funcionen en Docker y Local
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT_URL', 'http://127.0.0.1:9090')
MINIO_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'minio')
MINIO_SECRET = os.getenv('AWS_SECRET_ACCESS_KEY', 'minio123')
BUCKET_NAME = 'data'

# 2. CONFIGURACIÓN CRÍTICA PARA BOTO3 / MLFLOW
# Inyectamos las credenciales
os.environ['MLFLOW_S3_ENDPOINT_URL'] = MINIO_ENDPOINT
os.environ['AWS_ACCESS_KEY_ID'] = MINIO_KEY
os.environ['AWS_SECRET_ACCESS_KEY'] = MINIO_SECRET
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Configuración de MLflow (Tracking Server)

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', "http://127.0.0.1:5001")
EXPERIMENT_NAME = "Social_Network_Ads_Experiment"

# Opciones para Pandas (S3)
S3_OPTIONS = {
    'key': MINIO_KEY,
    'secret': MINIO_SECRET,
    'client_kwargs': {'endpoint_url': MINIO_ENDPOINT},
    'use_ssl': False
}

def load_data(filename):
    path = f's3://{BUCKET_NAME}/processed/{filename}'
    print(f"-> Leyendo: {path}")
    return pd.read_csv(path, storage_options=S3_OPTIONS)

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    """Entrena un modelo, evalúa y registra en MLflow"""
    
    # Iniciamos el run
    with mlflow.start_run(run_name=model_name, nested=True):
        print(f"\n--- Entrenando: {model_name} ---")
        
        # 1. Entrenar
        model.fit(X_train, y_train)
        
        # 2. Predecir
        y_pred = model.predict(X_test)
        
        # 3. Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"   Accuracy: {acc:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # 4. Logging en MLflow
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # 5. Registrar el modelo (Artifact)
        print(f"   -> Subiendo modelo a MinIO...")
        signature = infer_signature(X_train, y_pred)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        return acc, f1

def main():
    # 1. Configurar MLflow
    print(f"-> Conectando a MLflow en: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("-> Iniciando proceso de entrenamiento...")
    
    # 2. Cargar datos procesados
    try:
        train_df = load_data('train_scaled.csv')
        test_df = load_data('test_scaled.csv')
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return

    # Separar X e y
    TARGET = 'Purchased'
    X_train = train_df.drop(TARGET, axis=1)
    y_train = train_df[TARGET]
    X_test = test_df.drop(TARGET, axis=1)
    y_test = test_df[TARGET]
    
    # 3. Definir modelos a probar
    models = [
        ("Logistic_Regression", LogisticRegression(random_state=42)),
        ("Naive_Bayes_Gaussian", GaussianNB())
    ]
    
    best_model_name = None
    best_f1 = 0
    
    # 4. Iterar, entrenar y registrar
    for name, model in models:
        try:
            acc, f1 = train_and_log_model(model, name, X_train, y_train, X_test, y_test)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
        except Exception as e:
            print(f"❌ Error entrenando {name}: {e}")
            
            
    print(f"\n🏆 Mejor modelo: {best_model_name} (F1: {best_f1:.4f})")
    print("✅ Entrenamiento y registro en MLflow completado.")

if __name__ == "__main__":
    main()