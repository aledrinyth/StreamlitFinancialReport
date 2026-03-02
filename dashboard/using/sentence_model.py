### parses sentence by sentence until token limit reached, then runs FinBERT on each chunk, aggregates results.
# finbert_pipeline.py
import re
import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import hashlib

import pandas as pd
import numpy as np
# from diskcache import Cache

# Transformers imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

def sentence_split(text: str):
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


# ------------------------
# Config
# ------------------------
FINBERT_MODEL = "yiyanghkust/finbert-tone"  # common FinBERT; swap if you use another
MAX_TOKENS_FINBERT = 512
TARGET_TOKENS = 450   # leave margin for special tokens
NUM_WORKERS = max(1, os.cpu_count() - 1)
CACHE_DIR = "./cache_finbert"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
# cache = Cache(CACHE_DIR)
SAVE_EACH_RESULT = False  # whether to save each doc result separately



# ------------------------
# Load FinBERT model and tokenizer and pipeline
# ------------------------
print("Loading FinBERT model (this may take time)...")
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
finbert_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=-1)
print("FinBERT loaded.")

# ------------------------
# Utilities
# ------------------------
def sha1(s: str):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def sentence_split(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def tokens_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))

def chunk_sentences(sentences: List[str], max_tokens: int) -> List[str]:
    """
    Greedy pack sentences until token limit reached. Returns list of chunk strings.
    """
    chunks = []
    current = []
    current_tokens = 0
    for s in sentences:
        s_tokens = tokens_length(s)
        if s_tokens > max_tokens:
            # sentence alone too long -> optionally compress with LLM or split
            # for now, split by sub-sentences (fallback)
            parts = re.split(r'(?<=[\.\?\!])\s+', s)
            for p in parts:
                if p.strip():
                    if tokens_length(p) > max_tokens:
                        # force truncate (last resort) - better to call LLM here.
                        p = " ".join(p.split()[:max_tokens//2])
                    if tokens_length(" ".join(current + [p])) <= max_tokens:
                        current.append(p)
                    else:
                        if current:
                            chunks.append(" ".join(current))
                        current = [p]
            continue

        if current_tokens + s_tokens <= max_tokens:
            current.append(s)
            current_tokens += s_tokens
        else:
            if current:
                chunks.append(" ".join(current))
            current = [s]
            current_tokens = s_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

# ------------------------
# FinBERT inference wrapper
# ------------------------
def finbert_predict(text: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Returns probabilities vector [neg, neutral, pos] and a dict map label->prob
    """
    res = finbert_pipe(text[:10000])  # pipeline can fail on extremely long; ensure text not huge
    # res is a list of dicts for each label with 'label' and 'score'
    # mapping depends on model labels - let's map common labels to neg/neutral/pos
    # We'll sort by label names if they exist
    labels = {d['label'].lower(): d['score'] for d in res[0]}
    # Attempt common label names
    def get_prob(name):
        return labels.get(name, 0.0)
    # Known variants:
    neg = get_prob("negative") or get_prob("neg") or get_prob("bearish") or 0.0
    neu = get_prob("neutral") or get_prob("neutral/0") or 0.0
    pos = get_prob("positive") or get_prob("pos") or get_prob("bullish") or 0.0
    probs = np.array([neg, neu, pos], dtype=float)
    # normalize if needed
    if probs.sum() <= 0:
        probs = np.array([0.33, 0.34, 0.33], dtype=float)
    else:
        probs = probs / probs.sum()
    label_map = {"neg": float(probs[0]), "neu": float(probs[1]), "pos": float(probs[2])}
    return probs, label_map

# ------------------------
# Processing one document
# ------------------------
def process_single_document(text: str) -> Dict[str, Any]:
    """
    Process one report: chunk -> (optionally compress) -> FinBERT -> aggregate.
    Returns a dict with chunk-level and aggregated results.
    """
    #cache_key = f"res::{sha1(doc_id)}"
    #if cache_key in cache:
    #    return cache[cache_key]

    # Pre-clean: remove obvious disclaimers using regex (e.g., "disclaimer" blocks). Adjust as needed.
    text_clean = re.sub(r"(?is)^.*?disclaimer:.*?$", "", text)

    # Sentence split
    sentences = sentence_split(text_clean)

    # Chunk into token-safe groups
    chunks = chunk_sentences(sentences, max_tokens=TARGET_TOKENS)

    # If some chunks still exceed token length (rare), try compression
    final_chunks = []
    for ch in chunks:
        encoded = tokenizer.encode(ch, add_special_tokens=True)
        for i in range(0, len(encoded), TARGET_TOKENS):
            chunk_tokens = encoded[i:i+TARGET_TOKENS]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            final_chunks.append(chunk_text)

    # Run FinBERT on chunks
    chunk_results = []
    for ch in final_chunks :
        probs, label_map = finbert_predict(ch)
        chunk_results.append({
            "text": ch,
            "tokens": tokens_length(ch),
            "probs": probs.tolist(),
            "label_map": label_map
        })

    if not chunk_results:
        agg_probs = np.array([0.33, 0.34, 0.33])
    else:
        prob_matrix = np.array(
            [cr["probs"] for cr in chunk_results],
            dtype=float
        )
        agg_probs = prob_matrix.mean(axis=0)

    # Simple aggregated label (max prob)
    labels = ["neg", "neu", "pos"]
    agg_label = labels[int(np.argmax(agg_probs))]

    result = {
        "n_chunks": len(chunk_results),
        # "chunks": chunk_results,
        "agg_probs": agg_probs.tolist(),
        "agg_label": agg_label
    }
    if SAVE_EACH_RESULT:
        if file_name is not None:
            report_dir = Path("./results")
            result_path = report_dir / f"{file_name}_result.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


    # cache[cache_key] = result
    return result





# ------------------------
# Batch processing
# ------------------------
def process_documents_batch(docs: List[Tuple[str, str]], workers=NUM_WORKERS) -> List[Dict[str, Any]]:
    """
    docs: list of (doc_id, text)
    """
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_single_document, doc_id, text): doc_id for doc_id, text in docs}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                doc_id = futures[fut]
                print(f"Error processing {doc_id}: {e}")
    return results

# ------------------------
# Save results helpers
# ------------------------
def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "doc_id": r["doc_id"],
            "agg_label": r["agg_label"],
            "agg_neg": r["agg_probs"][0],
            "agg_neu": r["agg_probs"][1],
            "agg_pos": r["agg_probs"][2],
            "n_chunks": r["n_chunks"]
        })
    return pd.DataFrame(rows)

# ------------------------
# Example use
# ------------------------
if __name__ == "__main__":
    # Example: load all text files from ./reports and run pipeline
    report_dir = Path("./testing_reports")
    files = list(report_dir.glob("*.txt"))
    docs = []
    for f in files:
        file_name = f.stem
        doc_id = f.stem
        text = f.read_text(encoding="utf-8")
        docs.append((doc_id, text))

    # For demo, do sequential small-run (avoid spawning large processes in notebooks)
    results = []
    for doc_id, txt in tqdm(docs):
        results.append(process_single_document(doc_id, txt, file_name))

    df = results_to_dataframe(results)
    #df.to_parquet(Path(RESULTS_DIR) / "finbert_aggregated.parquet", index=False)
    #print("Done. Saved Parquet to:", Path(RESULTS_DIR) / "finbert_aggregated.parquet")
    df.to_csv(Path(RESULTS_DIR) / "finbert_aggregated.csv", index=False)
    print("Done. Saved CSV to:", Path(RESULTS_DIR) / "finbert_aggregated.csv")
