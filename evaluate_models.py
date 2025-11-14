import asyncio
import json
import time
from mcp_client_eval import get_mcp_answer      # import your corrected MCP wrapper
from rag.conversation_rag_eval import get_rag_answer # import your corrected RAG wrapper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
judge_llm = ChatGroq(model="qwen/qwen3-32b")  # same LLM for judge

def calc_answer_relevancy(question: str, answer: str) -> float:
    question_emb = embedding_model.encode([question])
    answer_emb = embedding_model.encode([answer])
    sim = cosine_similarity(question_emb, answer_emb)[0][0]
    return max(0.0, min(1.0, float(sim)))

def llm_score(prompt: str, key: str) -> float:
    response = judge_llm.invoke([HumanMessage(content=prompt)])
    try:
        data = json.loads(response.content.strip())
        return float(data.get(key, 0.0))
    except Exception as e:
        print("Parse error:", e)
        return 0.0

def calc_faithfulness(answer: str, context: str) -> float:
    prompt = f"""
You are an evaluator. Are all claims in the answer supported by context?
Context:
{context}
Answer:
{answer}
Instruction: Return ONLY JSON: {{"supported_claims": <int>, "total_claims": <int>, "score": <float>}}
"""
    return llm_score(prompt, "score")

def calc_context_relevancy(question: str, context: str) -> float:
    prompt = f"""
You are an evaluator. Rate context relevance for answering the question.
Question: {question}
Context: {context}
Instruction: Return ONLY JSON: {{"relevance_score": <float>}}
"""
    return llm_score(prompt, "relevance_score")

def calc_factual_correctness(answer: str, ground_truth: str) -> float:
    prompt = f"""
You are an evaluator. Compare answer and ground truth for factual overlap.
Answer: {answer}
Ground Truth: {ground_truth}
Instruction: Return ONLY JSON: {{"f1_score": <float>}}
"""
    return llm_score(prompt, "f1_score")

async def main():
    TESTS = [
        {
            "question": "What is machine learning?",
            "ground_truth": "Machine learning enables systems to learn from data without explicit programming."
        },
        {
            "question": "Explain neural networks.",
            "ground_truth": "Neural networks are computing systems inspired by biological neural networks."
        }
    ]

    for test in TESTS:
        print(f"\n=== Testing: {test['question']}")
        mcp_result = await get_mcp_answer(test["question"])
        rag_result = get_rag_answer(test["question"])

        # For MCP
        mcp_metrics = {
            "faithfulness": calc_faithfulness(mcp_result["answer"], mcp_result["context"]),
            "answer_relevancy": calc_answer_relevancy(test["question"], mcp_result["answer"]),
            "context_relevancy": calc_context_relevancy(test["question"], mcp_result["context"]),
            "factual_correctness": calc_factual_correctness(mcp_result["answer"], test["ground_truth"]),
            "response_time": mcp_result["response_time"]
        }
        print("MCP:", mcp_metrics)
        # For RAG
        rag_metrics = {
            "faithfulness": calc_faithfulness(rag_result["answer"], rag_result["context"]),
            "answer_relevancy": calc_answer_relevancy(test["question"], rag_result["answer"]),
            "context_relevancy": calc_context_relevancy(test["question"], rag_result["context"]),
            "factual_correctness": calc_factual_correctness(rag_result["answer"], test["ground_truth"]),
            "response_time": rag_result["response_time"]
        }
        print("RAG:", rag_metrics)

if __name__ == "__main__":
    asyncio.run(main())
