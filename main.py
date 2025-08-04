import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from app.langgraph_agent import build_graph
from app.vector_db import get_qdrant_client
from app.pdf_rag import ingest_pdf_to_vector_db
from app.llm_chain import process_pdf_response, process_weather_response
from app.eval_langsmith import evaluate_response_with_langsmith

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

st.title("AI Assistant: Weather & PDF Q&A")
user_query = st.text_input("Ask something…")

if st.button("Submit") and user_query:
    ctx = {
        "weather_api_key": OPENWEATHERMAP_API_KEY,
        "vector_db": ingest_pdf_to_vector_db(
            "NIPS-2017-attention-is-all-you-need-Paper.pdf",
            get_qdrant_client(QDRANT_URL, QDRANT_API_KEY),
            "my-pdf-collection"
        ),
        "openai_api_key": OPENAI_API_KEY
    }
    graph = build_graph()
    input_state = {
        "query": user_query,
        "context": ctx
    }
    raw_result = graph.invoke(input_state)

    # Decide which LLM post-processing based on returned data
    if "error" in raw_result:
        result = raw_result["error"]
    elif "docs" in raw_result:
        result = process_pdf_response(raw_result["docs"], user_query)
    else:
        result = process_weather_response(raw_result)

    st.markdown("### Response")
    st.write(result)

    evaluation = evaluate_response_with_langsmith(user_query, result)
    st.markdown("#### LangSmith Eval")
    st.json(evaluation)
