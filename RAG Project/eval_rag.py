import pandas as pd
import ast
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
 
model = SentenceTransformer("all-MiniLM-L6-v2")
 
 
def answer_relevancy(question, answer):
    q_emb = model.encode([question])
    a_emb = model.encode([answer])
    return cosine_similarity(q_emb, a_emb)[0][0]
 
 
def faithfulness(answer, contexts):
    if not contexts:
        return 0.0
 
    a_emb = model.encode([answer])
    c_embs = model.encode(contexts)
 
    sims = cosine_similarity(a_emb, c_embs)[0]
    return float(np.max(sims))
 
 
def context_recall(answer, contexts):
    if not contexts:
        return 0.0
 
    a_emb = model.encode([answer])
    c_embs = model.encode(contexts)
 
    sims = cosine_similarity(a_emb, c_embs)[0]
    return float((sims > 0.5).mean())
 
 
def evaluate_file(csv_path):
 
    df = pd.read_csv(csv_path)
 
    df["contexts"] = df["contexts"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
 
    results = []
 
    for _, row in df.iterrows():
 
        q = row["question"]
        a = row["answer"]
        c = row["contexts"]
 
        results.append({
            "answer_relevancy": answer_relevancy(q, a),
            "faithfulness": faithfulness(a, c),
            "context_recall": context_recall(a, c)
        })
 
    return pd.DataFrame(results).mean().to_dict()
 
 
methods = {
    "Hybrid": "scores_ragas_methode_1.csv",
    "Reranking": "scores_ragas_methode_2.csv",
    "HyDE": "scores_ragas_methode_3.csv"
}
 
all_results = []
 
for name, file in methods.items():
 
    print(f"Evaluating {name}...")
 
    scores = evaluate_file(file)
 
    all_results.append({
        "Method": name,
        **scores
    })
 
results_df = pd.DataFrame(all_results)
 
print("\n===== FINAL RESULTS =====")
print(results_df)
 
results_df.to_csv("rag_comparison_manual.csv", index=False)
 