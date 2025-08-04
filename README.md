# 🧠 AI Pipeline Assignment

This project demonstrates a modular AI pipeline integrating **LangChain**, **LangGraph**, **LangSmith**, and the **Qdrant vector database**, wrapped with a simple **Streamlit UI**.

---

## 🚀 Features

- 🔍 **Real-time Weather Queries** via OpenWeatherMap API
- 📄 **PDF Question Answering (RAG)** using LangChain + Qdrant
- 🧭 **Decision-making Agent** powered by LangGraph
- 🧠 **LLM Processing** and Summarization with LangChain
- 💾 **Embeddings and Vector Storage** using Qdrant
- 📊 **Evaluation** of LLM outputs with LangSmith
- 💬 **Chat Interface** using Streamlit
- 🧪 Modular and testable Python code

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Harshika-123/Weather-PDF-Insights-Agent.git
cd Weather-PDF-Insights-Agent
```

---

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# For Windows:
venv\Scripts\activate
# For macOS/Linux:
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add Environment Variables

Create a `.env` file in the project root with the following:

```
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Leave blank for local Qdrant
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_langsmith_project_name
LANGCHAIN_API_KEY=your_langsmith_api_key
```

> 🔑 Get your [OpenWeatherMap API key](https://openweathermap.org/appid)

---

### 5. Run Qdrant Vector Database

Run Qdrant locally via Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Or connect to a remote Qdrant instance by adjusting `QDRANT_URL` and `QDRANT_API_KEY`.

---

### 6. Add a Sample PDF

Add a PDF named `sample.pdf` to the project root directory for PDF Q&A.

---

## ▶️ Running the Application

Start the app:

```bash
streamlit run app/main.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Example Queries:

- `"What is the weather in New York?"` 🌤️
- `"Summarize the first section of the document."` 📄

---

## ✅ Running Tests

Run unit tests to verify components:

```bash
pytest tests/
```

---

## 📁 Project Structure

```
ai-pipeline-assignment/
│
├── app/
│   ├── main.py              # Streamlit chat interface
│   ├── langgraph_agent.py   # LangGraph decision agent
│   ├── weather.py           # Weather API logic
│   ├── pdf_rag.py           # PDF ingestion + RAG logic
│   ├── vector_db.py         # Qdrant client setup
│   ├── llm_chain.py         # LangChain LLM utilities
│   ├── eval_langsmith.py    # LangSmith evaluation
│   └── utils.py             # Utility functions
│
├── tests/                   # Unit tests
├── sample.pdf               # Sample document
├── requirements.txt         # Python package requirements
├── README.md                # Project documentation
└── .env                     # (Not committed) API keys and configs
```

---

## 📝 Notes

- Keep your API keys secure by using `.env` (don’t hardcode).
- LangGraph uses a Pydantic state model for flow control.
- Modify chunk size, model parameters, or node logic to fit your use case.
- LangSmith logs all evaluation steps for debugging and performance tracking.

---

## 🤝 Contributions

Feel free to open issues or submit PRs to improve the pipeline.

---

## 📬 Contact

For questions, support, or collaboration:
- 📧 your.email@example.com
- 🐙 GitHub Issues

---

Enjoy building with the AI pipeline! 🚀
