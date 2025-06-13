# LangGraph Family Tree QA

This project is a LangGraph-based system that answers questions about family relationships using a Prolog knowledge base and a retrieval-augmented generation (RAG) component. It uses GPT-4o-mini to interpret questions and decide whether to call a Prolog tool or retrieve relevant context using OpenAI embeddings. The knowledge base defines family facts and logic rules (like parent, grandparent), while the RAG tool summarizes those facts in natural language for flexible context matching. You can ask questions like “Who is the grandfather of Joe?” and the system will figure out how to answer it using either logic or context.

## How to Run

1. Create a `.env` file in the root of the project and add your OpenAI API key:

2. (Optional) Create and activate a virtual environment:

3. Install dependencies: pip install -r requirements.txt

4. Run the system, you should run main.py