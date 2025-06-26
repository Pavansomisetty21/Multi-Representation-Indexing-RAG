import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
docs = [
        Document(page_content="""
        This document explains the concept of memory in LLM-powered autonomous agents.
        It explores how agents retain and retrieve past interactions using vector databases,
        chunking strategies, and indexing methods. Several architectures like AutoGPT are analyzed.
        """),
        Document(page_content="""groq is an llm provider."""),

        Document(page_content="apslkxmcdlf,fdsunjkma,asdvufnjklzm")
    ]
class CustomSentenceTransformerEmbedder(Embeddings):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True)
    
embedding = CustomSentenceTransformerEmbedder()   
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatGroq(model="llama-3.1-8b-instant",api_key='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')
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})
from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

vectorstore = FAISS.from_documents(docs, embedding)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Docs linked to summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]
query="what are ai agents"
# Add
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))
retrieved_docs = retriever.get_relevant_documents(query,n_results=1)
print("=========multi represent rag answer==============")
retrieved_docs[0].page_content
