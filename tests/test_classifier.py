from app.retrieval.query_classifier import QueryClassifier


def test_classifier_routes_recent_query_to_rag():
    qc = QueryClassifier()
    route = qc.classify("What is the latest version of the policy?")
    assert route.route == "rag"
