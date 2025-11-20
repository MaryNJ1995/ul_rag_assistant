import streamlit as st

from ul_rag_assistant.ul_rag.interfaces.chat_session import RAGChatSession

st.set_page_config(page_title="UL RAG Assistant", page_icon="🎓")

st.title("UL RAG Assistant")
st.caption("University of Limerick RAG chatbot – by Maryam Najafi")

if "session" not in st.session_state:
    st.session_state["session"] = RAGChatSession(mode="student", locale="IE")

session: RAGChatSession = st.session_state["session"]

for turn in session.get_history():
    with st.chat_message(turn.role):
        st.markdown(turn.content)

user_input = st.chat_input("Ask something about UL...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_turn = session.ask(user_input)
            st.markdown(bot_turn.content)
