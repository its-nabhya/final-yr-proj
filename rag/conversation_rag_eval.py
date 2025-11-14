import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="assets/db/chroma_db", embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="qwen/qwen3-32b")

def get_rag_answer(question: str) -> dict:
    start_time = time.time()
    docs = retriever.invoke(question)
    # Compose context from top chunks
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        "You are a helpful assistant. Answer the following question using ONLY the information provided below:\n"
        f"Question: {question}\n"
        "Context:\n" + context +
        "\nIf the answer cannot be found in the provided context, say so."
    )
    messages = [
        SystemMessage(content="You are a helpful assistant answering based solely on provided context."),
        HumanMessage(content=prompt)
    ]
    result = llm.invoke(messages)
    answer = result.content.strip()
    end_time = time.time()
    return {
        "answer": answer,
        "context": context,
        "response_time": end_time - start_time
    }

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What is machine learning?"
    result = get_rag_answer(q)
    print("ANSWER:", result["answer"])
    print("CONTEXT:", result["context"])
    print("RESPONSE TIME:", result["response_time"])
