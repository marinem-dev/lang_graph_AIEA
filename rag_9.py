from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

@tool
def retrieve_context_rag(question: str) -> str:
    """
    Uses RAG to retrieve relevant info and answer a question based on NL descriptions of the Prolog KB.
    """
    raw_documents = """Tom is the father of Bob and Liz.
Bob is the father of Ann and Pat.
Pat is the parent of Jim.
Liz is the parent of Bill and Mary.
Mary is the parent of Joe and Sue."""

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents([Document(page_content=raw_documents)])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.from_documents(docs, embeddings)

    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=init_chat_model("gpt-4o-mini", model_provider="openai"), retriever=retriever)
    return qa.run(question)
