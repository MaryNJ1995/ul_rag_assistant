#!/usr/bin/env python
import streamlit as st

from ul_rag_assistant.ul_rag.interfaces.chat_session import RAGChatSession


APP_TITLE = "UL RAG Assistant"
APP_DESCRIPTION = (
    "Ask questions about the University of Limerick. "
    "The assistant uses a retrieval-augmented generation (RAG) pipeline over UL web content."
)


def get_session(mode: str, locale: str) -> RAGChatSession:
    """
    Create or reuse a chat session.

    If mode/locale changed compared to what's stored, we rebuild the session
    so the new settings take effect.
    """
    sess_key = "ul_rag_session"

    if sess_key not in st.session_state:
        st.session_state[sess_key] = RAGChatSession(mode=mode, locale=locale)
    else:
        sess: RAGChatSession = st.session_state[sess_key]
        # If the user changed the settings, recreate the session
        if getattr(sess, "mode", None) != mode or getattr(sess, "locale", None) != locale:
            st.session_state[sess_key] = RAGChatSession(mode=mode, locale=locale)

    return st.session_state[sess_key]


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.write(APP_DESCRIPTION)

    # ---------- Sidebar: Settings ----------
    with st.sidebar:
        st.header("Settings")

        # Mode selector: student / staff
        mode = st.radio(
            "Mode",
            options=["student", "staff"],
            index=0,
            help="Student mode is a bit more explanatory and friendly; staff mode is more concise.",
            key="ul_mode",
        )

        # Locale selector (you can extend this list if needed)
        locale = st.selectbox(
            "Locale",
            options=["IE", "EN", "EU"],
            index=0,
            help="Currently used mainly as a hint for language/formatting.",
            key="ul_locale",
        )

        # Button to reset the conversation
        if st.button("Reset conversation"):
            st.session_state.pop("ul_rag_session", None)
            st.experimental_rerun()

    # ---------- Main chat area ----------
    session = get_session(mode=mode, locale=locale)

    # Show existing history
    if session.history:
        for turn in session.history:
            with st.chat_message(turn.role):
                st.markdown(turn.content)

    # User input box
    user_input = st.chat_input("Ask me something about UL…")
    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Ask the RAG backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                bot_turn = session.ask(user_input)

            # Show answer
            st.markdown(bot_turn.content)

            # Optional: show citations
            if bot_turn.citations:
                with st.expander("Sources"):
                    for c in bot_turn.citations:
                        src = c.get("source", "unknown source")
                        st.markdown(f"- {src}")


if __name__ == "__main__":
    main()
