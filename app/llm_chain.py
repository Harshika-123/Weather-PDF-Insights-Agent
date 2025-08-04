from langchain.llms import OpenAI

def process_weather_response(raw_data):
    llm = OpenAI(temperature=0.2)
    prompt = (
        f"Summarize this weather information for a user: {raw_data}"
    )
    return llm(prompt)

def process_pdf_response(docs, query):
    llm = OpenAI(temperature=0.2)
    context_txt = "\n".join([doc.page_content for doc in docs])
    prompt = f"Answer this question based on the context:\n{context_txt}\nQuestion: {query}"
    return llm(prompt)
