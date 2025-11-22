# MLOps Final Project - Social Network Ads Pipeline

Este proyecto implementa un pipeline de **MLOps end-to-end** para la predicción de compras de usuarios basadas en publicidad en redes sociales. El sistema está completamente contenerizado utilizando **Docker** y orquesta el ciclo de vida del machine learning desde la ingesta de datos hasta el registro del modelo.

## 🏗️ Arquitectura del Proyecto

El entorno simula una infraestructura productiva real utilizando los siguientes servicios:

* **Apache Airflow:** Orquestador de flujos de trabajo (DAGs).
* **MinIO:** Data Lake (Object Storage compatible con S3) para almacenar datasets y artefactos.
* **MLflow:** Tracking server para el registro de experimentos, métricas y modelos.
* **PostgreSQL & Redis:** Backend y Broker para los servicios de orquestación.
* **Docker Compose:** Gestión de infraestructura como código.

---

## 🚀 Quick Start 

Este proyecto está configurado para desplegarse automáticamente con un solo comando, incluyendo la instalación de librerías y la carga del dataset.

### Prerrequisitos
* Docker Desktop instalado y corriendo.

### Ejecución
1.  Clonar el repositorio.
2.  Ejecutar el siguiente comando en la raíz del proyecto:

```bash
docker compose --profile all up -d --build
```

Esperar unos minutos a que los servicios indiquen estado healthy.

Acceso a los Servicios
Servicio	URL	Credenciales (User/Pass)
Airflow	http://localhost:8080	airflow / airflow
MinIO	http://localhost:9091	minio / minio123
MLflow	http://localhost:5001	N/A



## ⚙️ Flujo de Trabajo (Pipeline)
El DAG mlops_final_project automatiza los siguientes pasos:

1. Automatización de Infraestructura (docker-compose)
Se construye una imagen de Airflow personalizada inyectando el archivo requirements.txt.

Un contenedor efímero (create_s3_buckets) inicializa los buckets en MinIO y carga automáticamente el dataset Social_Network_Ads.csv.

2. Preparación de Datos (src/data_prep.py)
Ingesta: Lee el archivo crudo desde MinIO.

Limpieza: Elimina identificadores irrelevantes (User ID) y codifica variables categóricas (Gender).

Split: Divide el dataset en entrenamiento (80%) y prueba (20%).

Feature Scaling: Aplica StandardScaler solo a variables numéricas para evitar data leakage.

Artefactos: Guarda el objeto scaler.joblib en MinIO para su uso posterior en inferencia y los datasets procesados (train_scaled.csv, test_scaled.csv).

## 3. Entrenamiento y Selección (src/train.py)
Entrena múltiples modelos candidatos:

Regresión Logística.

Naive Bayes (Gaussian).

Evalúa el rendimiento utilizando métricas de Accuracy y F1-Score.

Utiliza MLflow para registrar parámetros, métricas y el modelo serializado.

Selecciona automáticamente el mejor modelo y lo promueve.

El modelo ganador queda registrado en MLflow listo para ser consumido.

## 🚧 Próximos Pasos (Roadmap)
El avance actual cubre la infraestructura, orquestación y entrenamiento. Para completar el ciclo de vida de MLOps productivo, los siguientes pasos están pendientes de implementación:

Despliegue de API (Serving):

Implementar el servicio con FastAPI (dockerfiles/fastapi/app.py).

Configurar el contenedor para descargar automáticamente el modelo "Champion" desde MLflow y el scaler desde MinIO al iniciarse.

Exponer el endpoint POST /predict para recibir datos de nuevos usuarios.

Monitoreo:

Implementar logs de predicción para detectar Data Drift en el futuro.

📂 Estructura del Repositorio
Plaintext
```
.
├── airflow/
│   └── dags/
│       └── pipeline.py      # Definición del DAG de Airflow
├── dockerfiles/             # Definiciones de imágenes Docker
├── src/
│   ├── data_prep.py         # Lógica de ETL y preprocesamiento
│   └── train.py             # Lógica de entrenamiento y MLflow
├── Social_Network_Ads.csv   # Dataset original
├── docker-compose.yaml      # Orquestación de servicios
├── requirements.txt         # Dependencias de Python
└── README.md                # Documentación del proyecto
```