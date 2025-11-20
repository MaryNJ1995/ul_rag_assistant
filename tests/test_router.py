from ul_rag.graph.router import Router


def test_router_default_plan(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = Router()
    plan = router.route("hi")
    assert plan.query_type in {"general", "chitchat", "nonsense"}
