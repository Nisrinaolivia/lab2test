import os
from pathlib import Path

import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

DATA_DIR = "lab2-starter-main"   # extracted folder from the zip
DB_DIR = "chroma_db"             # where the database is stored
COLLECTION = "canvassian"

# Key clients named in the README
CLIENTS = ["paywise", "alphabear", "bravocat", "charlemont", "deltaforce", "echona"]


def get_metadata(path):
    """Basic metadata from the file location: document type = folder name."""
    p = Path(path)
    return {
        "doc_type": p.parent.name,   # emails / contracts / board_papers
        "file_name": p.name,
    }


# 1. Load all .txt documents
print("Loading documents...")
documents = SimpleDirectoryReader(
    DATA_DIR,
    recursive=True,
    required_exts=[".txt"],
    file_metadata=get_metadata,
).load_data()
print(f"Number of documents: {len(documents)}")

# 2. Add metadata: which key clients are mentioned in each document
for doc in documents:
    haystack = (doc.text + " " + doc.metadata["file_name"]).lower()
    for client in CLIENTS:
        doc.metadata[f"mentions_{client}"] = client in haystack
    doc.metadata["mentions_jane_wu"] = "jane wu" in haystack

    # These flags are used for filtering only, so exclude them from embeddings
    doc.excluded_embed_metadata_keys = [k for k in doc.metadata if k.startswith("mentions_")]

# 3. Set up the embedding model and chunking
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# 4. Create (or reset) the ChromaDB collection
client = chromadb.PersistentClient(path=DB_DIR)
try:
    client.delete_collection(COLLECTION)   # start fresh on every run
except Exception:
    pass
collection = client.get_or_create_collection(COLLECTION)

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 5. Chunk, embed and store
print("Adding documents to ChromaDB (this may take a few minutes)...")
VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[splitter],
    show_progress=True,
)

print(f"Done! Number of chunks in the database: {collection.count()}")