import langsmith

def evaluate_response_with_langsmith(query, response):
    # Use your LangSmith setup. Visit LangSmith dashboard for API details.
    evaluation = langsmith.evaluate(query=query, completion=response)
    return evaluation
