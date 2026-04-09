from app.llm.reflection import ReflectionChecker


def test_reflection_returns_structured_output():
    checker = ReflectionChecker()
    out = checker.check("Paris is in France.", ["Paris is the capital of France."])
    assert "label" in out and "score" in out
