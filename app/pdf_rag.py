import PyPDF2
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qdrant_client.http.models import VectorParams, Distance
from qdrant_client.http.exceptions import UnexpectedResponse

def extract_pdf_text(pdf_path):
    reader = PyPDF2.PdfReader(open(pdf_path, "rb"))
    return " ".join([page.extract_text() for page in reader.pages])


def ingest_pdf_to_vector_db(pdf_path, qdrant_client, collection_name):
    # Define the vector size according to your embedding model
    vector_size = 1536  # e.g., OpenAI's embedding dimensionality

    # Check if collection exists by catching the 404 error
    try:
        qdrant_client.get_collection(collection_name)
    except UnexpectedResponse as e:
        # If 404 Not Found error for collection, create it
        # Check for "not found" by error message or status code
        if getattr(e, 'status_code', None) == 404 or 'not found' in str(e).lower():
            qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            # Re-raise if it's a different error
            raise e

    # Then proceed with text extraction, splitting and embedding...
    text = extract_pdf_text(pdf_path)
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

    embeddings = OpenAIEmbeddings()
    vector_db = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vector_db.add_texts(chunks)
    return vector_db

def query_pdf_rag(query, context):
    db = context["vector_db"]
    docs = db.similarity_search(query, k=3)
    return {"docs": docs, "query": query}
