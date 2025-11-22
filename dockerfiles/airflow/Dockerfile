FROM apache/airflow:3.0.2

USER root
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Volvemos al usuario airflow para instalar librerías de Python
USER airflow

# Copiamos el requirements.txt (que le pasaremos desde el contexto de la raíz)
COPY requirements.txt /requirements.txt

# Instalamos las dependencias
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt