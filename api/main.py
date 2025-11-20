from fastapi import FastAPI
from pydantic import BaseModel

from ul_rag.interfaces.chat_session import RAGChatSession

app = FastAPI(title="UL RAG Assistant API")

session = RAGChatSession(mode="student", locale="IE")


class Query(BaseModel):
    question: str
    mode: str = "student"
    locale: str = "IE"


@app.post("/chat")
def chat(q: Query):
    global session
    if q.mode != session.mode or q.locale != session.locale:
        session = RAGChatSession(mode=q.mode, locale=q.locale)
    turn = session.ask(q.question)
    return {
        "answer": turn.content,
        "citations": turn.citations,
        "meta": turn.meta,
    }
