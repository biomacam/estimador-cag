from __future__ import annotations

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")
st.caption("Describe your project and choose the level and format of the estimation.")

with st.form("estimation_form"):
    description = st.text_area("Project description", height=180, max_chars=2000)
    project_type = st.selectbox(
        "Project type",
        options=["mobile_app", "web_saas", "internal_tool", "data_pipeline"],
    )
    detail_level = st.selectbox("Detail level", options=["summary", "medium", "detailed"])
    output_format = st.selectbox(
        "Output format",
        options=["phases_table", "line_items", "narrative"],
    )
    submitted = st.form_submit_button("Estimate project")

if submitted:
    payload = {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    try:
        response = httpx.post(
            ESTIMATE_ENDPOINT,
            json=payload,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
        response.raise_for_status()
        body = response.json()
        st.markdown(body["text"])
        st.caption(f"Prompt version: {body['prompt_version']}")
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")

with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")