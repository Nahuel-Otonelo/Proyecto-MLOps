from airflow import DAG
from airflow.operators.bash import BashOperator
import pendulum
import os

default_args = {
    'owner': 'equipo_mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    'mlops_final_project',
    default_args=default_args,
    description='Pipeline de entrenamiento de Social Network Ads',
    # CAMBIO AQUÍ: Usamos 'schedule' en lugar de 'schedule_interval'
    schedule=None, 
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    tags=['mlops', 'classification'],
    catchup=False
) as dag:

    # --- CONFIGURACIÓN DE ENTORNO PARA DOCKER ---
    docker_env = {
        'MINIO_ENDPOINT_URL': 'http://minio:9000',
        'AWS_ACCESS_KEY_ID': 'minio',
        'AWS_SECRET_ACCESS_KEY': 'minio123',
        'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
        'AWS_DEFAULT_REGION': 'us-east-1'
    }

    # TAREA 1: Preparación de Datos
    data_prep_task = BashOperator(
        task_id='data_preparation',
        bash_command='python /opt/airflow/src/data_prep.py',
        env=docker_env 
    )

    # TAREA 2: Entrenamiento de Modelos
    train_model_task = BashOperator(
        task_id='model_training',
        bash_command='python /opt/airflow/src/train.py',
        env=docker_env
    )

    data_prep_task >> train_model_task