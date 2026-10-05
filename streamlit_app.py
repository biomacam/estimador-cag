from __future__ import annotations

import os
import threading
import time

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("ESTIMATOR_API_BASE_URL", "http://localhost:8000")
SESSIONS_ENDPOINT = f"{API_BASE_URL.rstrip('/')}/sessions"

# Same bound as the endpoint's transcript field.
MIN_TRANSCRIPT_CHARS = 20


def create_session() -> str | None:
    """POST /sessions; shows the error and returns None when it fails."""
    try:
        response = httpx.post(SESSIONS_ENDPOINT, timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        st.error(f"Could not create a session at `{SESSIONS_ENDPOINT}`: {exc}")
        return None
    return response.json()["session_id"]


def start_new_conversation() -> bool:
    """Replace the session and clear the result and the form (new widget keys)."""
    session_id = create_session()
    if session_id is None:
        return False
    st.session_state["session_id"] = session_id
    st.session_state.pop("result_body", None)
    st.session_state["form_version"] += 1
    return True


def error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        detail = body["detail"]
    except (ValueError, KeyError, TypeError):
        return response.text[:300]
    text = str(detail.get("message") or detail) if isinstance(detail, dict) else str(detail)
    reason = body.get("reason")
    if isinstance(reason, dict) and reason.get("code"):
        text += f" Reason: {reason['code']}"
        if reason.get("message"):
            text += f" ({reason['message']})"
    return text


def render_session_panel(session_id: str) -> None:
    """Sidebar: the memory (project metadata) shown apart from the history size."""
    st.header("Session")
    st.code(session_id, language="text")
    try:
        response = httpx.get(f"{SESSIONS_ENDPOINT}/{session_id}", timeout=5.0)
        response.raise_for_status()
        info = response.json()
    except httpx.HTTPError:
        st.caption("Session details unavailable.")
        return
    st.caption(f"History: {info['message_count']} messages (window of {info['max_turns']} turns)")
    st.subheader("Project metadata")
    st.json(info["metadata"])


st.set_page_config(page_title="Software Estimator", page_icon="📊")
st.title("Software Estimator")

st.session_state.setdefault("form_version", 0)

if "session_id" not in st.session_state:
    new_session_id = create_session()
    if new_session_id is None:
        st.stop()
    st.session_state["session_id"] = new_session_id
session_id = st.session_state["session_id"]
ESTIMATE_ENDPOINT = f"{SESSIONS_ENDPOINT}/{session_id}/estimate"

# Bumping form_version changes every widget's key below, so Streamlit creates
# brand-new, empty widgets instead of reusing values tied to the old keys.
if st.button("🆕 Nueva conversación", key="new_conversation"):
    if start_new_conversation():
        st.rerun()

st.caption(
    "Describe your project and attach supporting documents. Every message continues the "
    "same conversation, so the service remembers what was agreed before."
)

form_version = st.session_state["form_version"]
with st.form("estimation_form"):
    transcript = st.text_area(
        "Transcript", height=180, max_chars=50_000, key=f"transcript_{form_version}",
    )
    attachments = st.file_uploader(
        "Attachments (PDF or Word)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key=f"attachments_{form_version}",
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
    submitted = st.form_submit_button("Estimate project", key="estimate")

if submitted and len(transcript.strip()) < MIN_TRANSCRIPT_CHARS:
    st.error(f"The transcript must be at least {MIN_TRANSCRIPT_CHARS} characters long.")
elif submitted:
    form_data = {
        "transcript": transcript,
        "project_type": project_type,
        "detail_level": detail_level,
        "output_format": output_format,
    }
    files = [
        ("attachments", (upload.name, upload.getvalue(), upload.type or "application/octet-stream"))
        for upload in attachments
    ]

    # The POST call blocks for up to a couple of minutes (estimation plus the metadata
    # extraction), so it runs on a background thread while this loop polls the elapsed
    # time to keep the counter below live.
    status_placeholder = st.empty()
    counter_placeholder = st.empty()
    call_state: dict = {}

    def _call_estimator() -> None:
        try:
            call_state["response"] = httpx.post(
                ESTIMATE_ENDPOINT,
                data=form_data,
                files=files or None,
                timeout=httpx.Timeout(180.0, connect=10.0),
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
            f"Esperando respuesta estructurada (puede tardar un par de minutos)... "
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
        if response.is_success:
            st.session_state["result_body"] = response.json()
            # New widget keys clear the transcript and the uploads for the next turn.
            st.session_state["form_version"] += 1
            st.rerun()
        elif response.status_code == 404:
            st.error(
                "This session no longer exists (the service was restarted). "
                "Start a new conversation."
            )
        else:
            st.error(f"The estimator returned {response.status_code}: {error_detail(response)}")

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
    render_session_panel(session_id)
    st.header("Service")
    st.code(SESSIONS_ENDPOINT, language="text")
    primary = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
    fallback = os.getenv("FALLBACK_MODEL", "claude-haiku-4-5-20251001")
    st.markdown(f"**Primary model:** `{primary}`")
    st.markdown(f"**Fallback model:** `{fallback}`")
    st.markdown(f"**Cache TTL:** `{os.getenv('CACHE_TTL', '86400')}s`")