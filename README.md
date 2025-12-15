# MLOps Final Project - Social Network Ads Pipeline

Este proyecto implementa un pipeline de **MLOps end-to-end** para la predicción de compras de usuarios basadas en publicidad en redes sociales. El sistema está completamente contenerizado utilizando **Docker** y orquesta el ciclo de vida del machine learning desde la ingesta de datos hasta el despliegue del modelo via API.

## 🏗️ Arquitectura del Proyecto

El entorno simula una infraestructura productiva real utilizando los siguientes servicios:

* **Apache Airflow:** Orquestador de flujos de trabajo (DAGs).
* **MinIO:** Data Lake (Object Storage compatible con S3) para almacenar datasets y el artefacto `scaler`.
* **MLflow:** Tracking server para el registro de experimentos y gestión de modelos.
* **FastAPI:** Microservicio REST para servir predicciones en tiempo real (Inferencia).
* **PostgreSQL & Redis:** Backend y Broker para los servicios de orquestación.
* **Docker Compose:** Gestión de infraestructura como código.

---

## 🚀 Quick Start 

Este proyecto utiliza variables de entorno para la gestión segura de credenciales.

### Prerrequisitos
1.  **Docker Desktop** instalado y corriendo.
2.  Crear un archivo **`.env`** en la raíz del proyecto (ver sección Configuración).

### Ejecución
1.  Clonar el repositorio.
2.  Ejecutar el siguiente comando en la raíz del proyecto para construir y levantar los servicios:

```bash
docker compose --profile all up -d --build
```
Esperar unos minutos a que los servicios indiquen estado healthy.

### 🔌 Acceso a los Servicios

| Servicio | URL | Credenciales (User/Pass) |
| :--- | :--- | :--- |
| **Airflow** | http://localhost:8080 | `airflow` / `airflow` |
| **MinIO** | http://localhost:9091 | `minio` / `minio123` |
| **MLflow** | http://localhost:5001 | N/A |
| **FastAPI** (Docs) | http://localhost:8800/docs | N/A |

### 🔐 Configuración y Seguridad (.env)

El proyecto sigue buenas prácticas de DevSecOps, evitando credenciales hardcodeadas. Antes de iniciar, asegúrate de tener el archivo `.env` en la raíz con la siguiente configuración:

```ini
# Configuración Básica
AIRFLOW_UID=50000
MINIO_ACCESS_KEY=minio
MINIO_SECRET_ACCESS_KEY=minio123
DATA_REPO_BUCKET_NAME=data

# Endpoints Internos
MLFLOW_S3_ENDPOINT_URL=http://s3:9000
```

## ⚙️ Flujo de Trabajo (Pipeline)

El sistema automatiza el ciclo completo de MLOps:

1.  **Infraestructura (Docker Compose)**
    *   Despliegue automático de servicios, creación de buckets en MinIO y carga inicial del dataset crudo.
2.  **Preparación de Datos (Airflow: `data_prep.py`)**
    *   **Ingesta:** Lee el archivo crudo desde MinIO.
    *   **Limpieza:** Codifica variables categóricas (Gender).
    *   **Split & Scaling:** Divide en Train/Test y aplica `StandardScaler`.
    *   **Artefactos:** Guarda el `scaler.joblib` en MinIO (`artifacts/`) para asegurar consistencia en la inferencia.
3.  **Entrenamiento y Selección (Airflow: `train.py`)**
    *   Entrena modelos candidatos (Regresión Logística, Naive Bayes).
    *   Registra métricas y parámetros en MLflow.
    *   Selecciona el mejor modelo y lo registra en el Model Registry.
4.  **Despliegue e Inferencia (FastAPI)**
    *   El contenedor `fastapi_model_serving` expone el modelo seleccionado:
        *   **Arranque Seguro:** Valida la existencia de variables de entorno críticas; si faltan, el servicio se detiene (Fail-fast).
        *   **Carga de Artefactos:** Descarga el scaler desde MinIO y el modelo desde MLflow usando `boto3`.
        *   **Validación de Input:** Normaliza entradas (e.g., "Male", "male", "MALE") y rechaza valores inválidos (Error 400).
        *   **Endpoint:** `POST /predict` devuelve la predicción de compra.

## 🧪 Ejemplo de Uso (API)

Puedes probar la API desde la interfaz Swagger UI o vía curl:

**Request (JSON):**

```json
{
  "Gender": "Female",
  "Age": 45,
  "EstimatedSalary": 90000
}
```

**Respuesta Esperada:**

```json
{
  "prediction": 1,
  "label": "COMPRA"
}
```

## 📂 Estructura del Repositorio

```plaintext
.
├── airflow/                 # DAGs y Configuración
├── dockerfiles/             
│   └── fastapi/             # Código de la API, Dockerfile y requirements
├── src/                     # Lógica de ETL y Entrenamiento
├── .env                     # Variables de entorno (No incluido en git)
├── .dockerignore            # Optimización de build
├── docker-compose.yaml      # Orquestación de servicios
└── README.md                # Documentación
```