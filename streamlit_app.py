#Hace que las anotaciones de tipos se traten de una forma más flexible
from __future__ import annotations
#leer variables de entorno.
import os
#Iterator: expresar que la función va entregando varios textos progresivamente
from collections.abc import Iterator
#se comunica con FastAPI mediante HTTP
import httpx
import streamlit as st
#cargar variables de entorno desde el archivo .env
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
STREAM_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate/stream"

#dibujamos el título de la página
st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption(
    "Paste a meeting transcription. The answer streams token by token from the "
    "FastAPI service over Server-Sent Events."
)