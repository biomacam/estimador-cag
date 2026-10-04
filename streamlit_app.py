from __future__ import annotations

import os
import threading
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
ESTIMATE_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/api/v1/estimate"

st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")

st.session_state.setdefault("form_version", 0)

# Mirrors estimator-web's "Nueva estimación" link: clears the form and the
# last result so the user starts from a blank screen, without a page reload.
# Bumping form_version changes every widget's key below, so Streamlit creates
# brand-new, empty widgets instead of reusing values tied to the old keys.
if st.session_state.get("result_body") is not None:
    if st.button("🆕 Nueva estimación"):
        st.session_state.pop("result_body", None)
        st.session_state["form_version"] += 1
        st.rerun()

st.caption("Describe your project and choose the level and format of the estimation.")

form_version = st.session_state["form_version"]
with st.form("estimation_form"):
    description = st.text_area(
        "Project description", height=180, max_chars=2000, key=f"description_{form_version}",
    )
    project_type = st.selectbox(
        "Project type",
        options=["mobile_app", "web_saas", "internal_tool", "data_pipeline"],
        key=f"project_type_{form_version}",
    )
    detail_level = st.selectbox(
        "Detail level", options=["summary", "medium", "detailed"], key=f"detail_level_{form_version}",
    )
    output_format = st.selectbox(
        "Output format",
        options=["phases_table", "line_items", "narrative"],
        key=f"output_format_{form_version}",
    )
    submitted = st.form_submit_button("Estimate project")

if submitted:
    payload = {
        "description": description,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }

    # The POST call blocks for up to ~90s, so it runs on a background thread
    # while this loop polls the elapsed time to keep the counter below live.
    status_placeholder = st.empty()
    counter_placeholder = st.empty()
    call_state: dict = {}

    def _call_estimator() -> None:
        try:
            call_state["response"] = httpx.post(
                ESTIMATE_ENDPOINT,
                json=payload,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        except httpx.HTTPError as exc:
            call_state["error"] = exc

    thread = threading.Thread(target=_call_estimator, daemon=True)
    start_time = time.monotonic()
    thread.start()

    status_placeholder.markdown(
        """
        <span style="background-color:#e8b923; color:#1a1a1a; padding:4px 14px;
        border-radius:14px; font-weight:600; font-size:0.85rem;">
        &#9203; Generating...</span>
        """,
        unsafe_allow_html=True,
    )
    while thread.is_alive():
        elapsed = int(time.monotonic() - start_time)
        counter_placeholder.markdown(
            f"Esperando respuesta estructurada (puede tardar hasta 90 s)... "
            f"`{elapsed // 60:02d}:{elapsed % 60:02d}`"
        )
        time.sleep(1)
    thread.join()

    status_placeholder.empty()
    counter_placeholder.empty()

    if "error" in call_state:
        st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {call_state['error']}")
    else:
        response = call_state["response"]
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            st.error(f"Could not reach the estimator at `{ESTIMATE_ENDPOINT}`: {exc}")
        else:
            body = response.json()
            st.session_state["result_body"] = body

result_body = st.session_state.get("result_body")
if result_body is not None:
    badges = (
        f'<span style="background-color:#e8b923; color:#1a1a1a; padding:3px 10px; '
        f'border-radius:12px; font-weight:700; font-size:0.75rem; margin-right:6px;">'
        f'PROMPT {result_body["prompt_version"].upper()}</span>'
    )
    if result_body.get("cached"):
        badges += (
            '<span style="background-color:#2ea043; color:#ffffff; padding:3px 10px; '
            'border-radius:12px; font-weight:700; font-size:0.75rem;">CACHED</span>'
        )
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown(result_body["text"])

with st.sidebar:
    st.header("Service")
    st.code(ESTIMATE_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")