import json
import time
import requests
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent / "data" / "ground_truth_questions.json"
API_URL = "http://127.0.0.1:8000/api/benchmark"

def run_evaluation():
    if not DATA_FILE.exists():
        print(f"Error: Could not find {DATA_FILE}")
        return

    with DATA_FILE.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions from dataset.")
    print("Sending batch to MedGraph AI benchmark endpoint (this will take several minutes)...")
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json={"items": questions})
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to backend or run benchmark: {e}")
        return

    elapsed = time.time() - start_time
    print(f"\n✅ Evaluation complete in {elapsed:.1f} seconds.\n")

    print("--- BENCHMARK RESULTS ---")
    print(f"Evaluated: {results.get('items_evaluated')} items")
    print(f"Generation Providers: {results.get('generation_providers')}")
    print(f"Judge Provider: {results.get('judge_provider')} (Gemini API)\n")

    for pipeline, data in results.get("aggregate", {}).items():
        print(f"Pipeline: {pipeline.upper()}")
        print(f"  Avg Tokens:  {data['tokens']}")
        print(f"  Avg Latency: {data['latency_ms']} ms")
        print(f"  Cost (USD):  ${data['cost_usd']:.6f}")
        
        # Accuracy might be None if LLM judge fails or is disabled
        acc = data.get('accuracy')
        if acc is not None:
            print(f"  Accuracy:    {acc}% (Gemini LLM-as-a-judge)")
        else:
            print("  Accuracy:    Judge unavailable")
        bert = data.get("bertscore_f1")
        if bert is not None:
            print(f"  BERTScore:   {bert:.4f}")
        else:
            print("  BERTScore:   Unavailable")
        print()

    improvements = results.get("improvements") or {}
    print("--- GRAPHRAG IMPROVEMENTS VS BASIC RAG ---")
    print(f"Token reduction: {improvements.get('graph_token_reduction_vs_rag_percent')}%")
    print(f"Cost reduction:  {improvements.get('graph_cost_reduction_vs_rag_percent')}%")
    print(f"Accuracy lift:   {improvements.get('graph_accuracy_improvement_vs_rag_percent')}%")

    print("Note: The MedGraph AI backend automatically evaluates 'PASS/FAIL' using Gemini instead of Hugging Face, as configured in medgraph/config.py.")

if __name__ == "__main__":
    run_evaluation()
