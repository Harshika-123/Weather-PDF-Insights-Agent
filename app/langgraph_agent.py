from langgraph.graph import StateGraph
from app.weather import fetch_weather
from app.pdf_rag import query_pdf_rag

from pydantic import BaseModel
from typing import Dict

def is_weather_query(user_query: str) -> bool:
    keywords = ["weather", "temperature", "forecast", "humidity", "rain"]
    return any(word in user_query.lower() for word in keywords)

def agent_node(state: Dict):
    query = state.query
    context = state.context
    if is_weather_query(query):
        return fetch_weather(query, context)
    else:
        return query_pdf_rag(query, context)

# Define the graph state as a Pydantic model
class StateSchema(BaseModel):
    query: str
    context: Dict

def build_graph():
    g = StateGraph(state_schema=StateSchema)
    g.add_node("main", agent_node)
    g.set_entry_point("main")  # safer than directly assigning start_node
    return g.compile()
