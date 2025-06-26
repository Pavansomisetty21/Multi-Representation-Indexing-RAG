import uuid
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.embeddings.base import Embeddings

class CustomSentenceTransformerEmbedder(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True)

def main():
    embedding = CustomSentenceTransformerEmbedder()

    docs = [
        Document(page_content="""
        This document explains the concept of memory in LLM-powered autonomous agents.
        It explores how agents retain and retrieve past interactions using vector databases,
        chunking strategies, and indexing methods. Several architectures like AutoGPT are analyzed.
        """),
        Document(page_content="""groq is an llm provider.""")
    ]

    doc_ids = [str(uuid.uuid4()) for _ in docs]

    
    for doc, doc_id in zip(docs, doc_ids):
        doc.metadata["doc_id"] = doc_id

    db = FAISS.from_documents(docs, embedding)

   
    docstore = InMemoryStore()
    docstore.mset(list(zip(doc_ids, docs)))

    
    retriever = MultiVectorRetriever(
        vectorstore=db,
        docstore=docstore,
        id_key="doc_id"
    )

    
    query = "groq"
    retrieved_docs = retriever.invoke(query, n_results=1)
    sub_docs = db.similarity_search(query, k=1)

    print("\n\n\nSimilarity Search from FAISS:\n", sub_docs)

    if retrieved_docs:
        print("\n\nRetrieved document(s) from MultiVectorRetriever:\n")
        for d in retrieved_docs:
            print(d.page_content)
    else:
        print("No documents retrieved.")

if __name__ == "__main__":
    main()
