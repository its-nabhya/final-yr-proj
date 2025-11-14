# # rag/conversation_rag.py
# from dotenv import load_dotenv
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
# from langchain.chains import RetrievalQA
# from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# # Load persisted vectorstore
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# db = Chroma(
#     persist_directory="assets/db/chroma_db",
#     embedding_function=embedding_model
# )
# retriever = db.as_retriever(search_kwargs={"k": 3})

# # Choose your LLM (Ollama, OpenAI, or Gemini)
# llm = Ollama(model="llama3.2:3b")  # Or whatever you have running

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=retriever,
#     return_source_documents=True,
# )

# # Rudimentary chat history system
# chathistory = []

# def ask_question(user_question):
#     print(f"--- You asked: {user_question} ---")
#     if chathistory:
#         # Rewrites question based on context (optionally)
#         prev_msgs = [HumanMessage(content=q) if i % 2 == 0 else AIMessage(content=q) for i, q in enumerate(chathistory)]
#         rewritten = llm.invoke([SystemMessage(content="Rewrite as standalone:"), *prev_msgs, HumanMessage(content=user_question)]).strip()
#         search_question = rewritten
#     else:
#         search_question = user_question

#     docs = retriever.invoke(search_question)
#     context = "\n".join([doc.page_content for doc in docs])
#     combined_input = f"Based ONLY on these docs, answer: {user_question}\nDocs:\n{context}"
#     answer = llm.invoke([SystemMessage(content="Answer based on the given docs. If unsure, say so."), HumanMessage(content=combined_input)])

#     print(f"Context preview:\n{context[:400]}...")
#     print(f"Answer: {answer}")
#     chathistory.append(user_question)
#     chathistory.append(answer)
#     return answer

# # Interactive CLI
# if __name__ == "__main__":
#     print("Ask questions! Type 'quit' to exit.")
#     while True:
#         q = input("Q: ")
#         if q.strip().lower() == 'quit':
#             print("Bye!")
#             break
#         ask_question(q)
# rag/conversation_rag.py

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq  # ✅ Use Groq API integration
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Set up embeddings and database
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory="assets/db/chroma_db",
    embedding_function=embedding_model
)

# Set up retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# ✅ Use Groq model
llm = ChatGroq(model="qwen/qwen3-32b")  # main model

# Chat history
chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Rewrite question (if context exists)
    if chat_history:
        rewrite_messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        rewrite_result = llm.invoke(rewrite_messages)
        search_question = rewrite_result.content.strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Retrieve documents
    docs = retriever.invoke(search_question)
    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        preview = "\n".join(doc.page_content.split("\n")[:2])
        print(f"  Doc {i}: {preview}...")

    # Step 3: Build input for final answer
    combined_input = f"""You are a helpful assistant. Answer the user's question based ONLY on the information from these documents.

Question: {user_question}

Documents:
{"".join(f"- {doc.page_content}\n" for doc in docs)}

If the answer cannot be found in the provided documents, say: "I don't have enough information to answer that question based on the provided documents."
"""

    # Step 4: Get answer
    messages = [
        SystemMessage(content="You are a helpful assistant answering based on the provided context."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = llm.invoke(messages)
    answer = result.content.strip()

    print(f"\nAnswer: {answer}\n")

    # Step 5: Update chat history
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    return answer


# Simple CLI loop
def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    while True:
        user_input = input("\nYour question: ").strip()
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        ask_question(user_input)


def ask_question_eval(user_question):
    print(f"--- You asked: {user_question} ---")
    # Rewrite if needed for context
    search_question = user_question
    docs = retriever.invoke(search_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"Found {len(docs)} relevant documents.")

    combined_input = (
        f"You are a helpful assistant. Answer the user's question based only on the information from these documents.\n"
        f"Question: {user_question}\n"
        f"Documents:\n"
        + "\n---\n".join([doc.page_content for doc in docs])
        + "\nIf the answer cannot be found in the provided documents, say so."
    )

    messages = [
        SystemMessage(content="You are a helpful assistant answering based on the provided context."),
        HumanMessage(content=combined_input)
    ]
    result = llm.invoke(messages)
    answer = result.content.strip()
    print(answer)
    # For evaluation: return both answer and context
    return {
        "answer": answer,
        "context": context
    }


if __name__ == "__main__":
    # start_chat()
    user_question = input("Ask a question: ")
    result = ask_question_eval(user_question)
    print("ANSWER:", result["answer"])
    print("CONTEXT:", result["context"])
