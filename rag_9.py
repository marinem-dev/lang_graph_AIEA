from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


@tool
def retrieve_context_rag(question: str) -> str:
    """
    Retrieves the most relevant context from embedded knowledge base sentences.
    """
    raw_sentences = [
        "Tom is the father of Bob and Liz.",
        "Bob is the father of Ann and Pat.",
        "Pat is the parent of Jim.",
        "Liz is the parent of Bill and Mary.",
        "Mary is the parent of Joe and Sue."
    ]

    docs = [Document(page_content=sent) for sent in raw_sentences]

    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    split_docs = splitter.split_documents(docs)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set."
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

    vectordb = FAISS.from_documents(split_docs, embeddings)
    retriever = vectordb.as_retriever()

    results = retriever.invoke(question)
    if isinstance(results, list) and results:
        return f"Retrieved context: {results[0].page_content}"
    return "No relevant context found."

if __name__ == "__main__":
    result = retrieve_context_rag("Who is the father of Pat?")
    print("Result:", result)