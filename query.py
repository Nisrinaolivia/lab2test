import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

DB_DIR = "chroma_db"
COLLECTION = "canvassian"
LLM_MODEL = "gpt-4o-mini"

# Same embedding model as ingest.py (must match!)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.llm = OpenAI(model=LLM_MODEL, temperature=0)

# Load the existing ChromaDB collection (no re-embedding needed)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store)

# Instructions for the LLM: act as an M&A due diligence lawyer
QA_PROMPT = PromptTemplate(
    "You are a lawyer conducting due diligence for a purchaser acquiring "
    "Canvassian Pty Ltd, a cybersecurity software company.\n"
    "Answer the question using ONLY the document excerpts below. "
    "If the excerpts do not contain the answer, say so clearly. "
    "Quote key wording where relevant and name the source file for each finding.\n\n"
    "Document excerpts:\n{context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer:"
)


def ask(question, doc_type=None, client_name=None, top_k=8):
    """Search the database and answer a question, with optional metadata filters."""
    # Build a ChromaDB "where" filter (supports True/False values)
    conditions = []
    if doc_type:
        conditions.append({"doc_type": doc_type})
    if client_name:
        conditions.append({f"mentions_{client_name.lower()}": True})

    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}
    else:
        where = None

    query_engine = index.as_query_engine(
        similarity_top_k=top_k,
        vector_store_kwargs={"where": where} if where else {},
        text_qa_template=QA_PROMPT,
    )
    response = query_engine.query(question)

    print("=" * 80)
    print(f"QUESTION: {question}")
    if where:
        print(f"FILTERS: {where}")
    print("-" * 80)
    print(response.response)
    print("\nSOURCES:")
    for node in response.source_nodes:
        print(f"  - {node.metadata.get('file_name')} (score: {node.score:.3f})")
    print()
    return response


if __name__ == "__main__":
    # Quick tests, one for each critical risk
    ask("Is there any indication that the founder Jane Wu may leave the company "
        "or is losing motivation to lead it?")

    ask("Is PayWise experiencing financial difficulties?", client_name="paywise")

    ask("Does this contract contain a change of control clause? "
        "What happens if Canvassian is acquired?",
        doc_type="contracts", client_name="bravocat")

