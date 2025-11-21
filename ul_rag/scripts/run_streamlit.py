#!/usr/bin/env python
import os
import json
from datetime import datetime
import uuid

import streamlit as st

from ul_rag_assistant.ul_rag.interfaces.chat_session import RAGChatSession


APP_TITLE = "UL RAG Assistant"
APP_DESCRIPTION = (
    "Ask questions about the University of Limerick. "
    "This assistant uses a retrieval-augmented generation (RAG) pipeline over UL content."
)

LOG_DIR = "/home/maryam_najafi/ul_bot/ul_rag_assistant/logs"
LOG_FILE = os.path.join(LOG_DIR, "streamlit_chats.jsonl")


# ---------- Logging helper ----------

def append_chat_log(
    session: RAGChatSession,
    user_text: str,
    bot_content: str,
    bot_citations,
) -> None:
    """
    Append two records (user + assistant) to a JSONL log file.

    Each line is a JSON object with:
      - session_id
      - timestamp
      - role ("user" / "assistant")
      - mode, locale
      - content
      - citations (for assistant turns)
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.utcnow().isoformat() + "Z"

    base = {
        "session_id": session.session_id,
        "timestamp": ts,
        "mode": session.mode,
        "locale": session.locale,
    }

    user_record = {
        **base,
        "role": "user",
        "content": user_text,
        "citations": [],
    }

    bot_record = {
        **base,
        "role": "assistant",
        "content": bot_content,
        "citations": bot_citations or [],
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(user_record, ensure_ascii=False) + "\n")
        f.write(json.dumps(bot_record, ensure_ascii=False) + "\n")


# ---------- Streamlit helpers ----------

def get_session(mode: str, locale: str) -> RAGChatSession:
    """
    Create or reuse a chat session.

    If mode/locale changed, rebuild the session so settings take effect.
    """
    sess_key = "ul_rag_session"
    sid_key = "ul_rag_session_id"

    # Allocate a per-browser-session id
    if sid_key not in st.session_state:
        st.session_state[sid_key] = str(uuid.uuid4())

    session_id = st.session_state[sid_key]

    if sess_key not in st.session_state:
        st.session_state[sess_key] = RAGChatSession(
            mode=mode,
            locale=locale,
            session_id=session_id,
        )
    else:
        sess: RAGChatSession = st.session_state[sess_key]
        if getattr(sess, "mode", None) != mode or getattr(sess, "locale", None) != locale:
            # reset session but keep same session_id
            st.session_state[sess_key] = RAGChatSession(
                mode=mode,
                locale=locale,
                session_id=session_id,
            )

    return st.session_state[sess_key]


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Settings")

        mode = st.radio(
            "Mode",
            options=["student", "staff"],
            index=0,
            key="ul_mode",
        )

        locale = st.selectbox(
            "Locale",
            options=["IE", "EN", "EU"],
            index=0,
            key="ul_locale",
        )

        if st.button("Reset conversation"):
            st.session_state.pop("ul_rag_session", None)
            st.experimental_rerun()

        st.markdown("---")
        st.subheader("Logs")

        # Optional: simple check to see if log file exists
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                log_data = f.read()
            st.download_button(
                label="Download chat logs (JSONL)",
                data=log_data,
                file_name="streamlit_chats.jsonl",
                mime="application/json",
            )
        else:
            st.caption("No logs yet. Start chatting to create log entries.")

    # ---------- Main chat ----------
    session = get_session(mode=mode, locale=locale)

    # Show previous turns
    if session.history:
        for turn in session.history:
            with st.chat_message(turn.role):
                st.markdown(turn.content)

    # Input
    user_input = st.chat_input("Ask me something about UL…")

    if user_input:
        # show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # get bot answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                bot_turn = session.ask(user_input)

            st.markdown(bot_turn.content)

            if bot_turn.citations:
                with st.expander("Sources"):
                    for c in bot_turn.citations:
                        src = c.get("source", "unknown source")
                        st.markdown(f"- {src}")

            # ---- log this exchange ----
            append_chat_log(
                session=session,
                user_text=user_input,
                bot_content=bot_turn.content,
                bot_citations=bot_turn.citations,
            )


if __name__ == "__main__":
    main()
