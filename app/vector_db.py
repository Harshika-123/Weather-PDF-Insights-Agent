from qdrant_client import QdrantClient

def get_qdrant_client(url, api_key):
    return QdrantClient(url=url, api_key=api_key)
