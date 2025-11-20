from ul_rag.graph.graph import run_ul_rag


def test_run_ul_rag_shape():
    resp = run_ul_rag("When do spring exams start?")
    assert isinstance(resp, dict)
    assert "answer" in resp
    assert "citations" in resp
    assert "meta" in resp
    assert "plan" in resp
