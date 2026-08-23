"""Evaluate the retained v7 models on etiketli_veri_seti.json."""
import json

import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("Veri yükleniyor...", flush=True)
data = json.load(open("etiketli_veri_seti.json"))

comp_texts, comp_true = [], []
rel_pairs, rel_true = [], []
for doc in data:
    sid2text = {s["id"]: s["text"] for s in doc["sentences"]}
    for s in doc["sentences"]:
        comp_texts.append(s["text"])
        comp_true.append(s["label"])
    for r in doc["relations"]:
        src = sid2text.get(r["src"], "")
        dst = sid2text.get(r["dst"], "")
        if src and dst:
            rel_pairs.append((src, dst))
            rel_true.append(r["type"])

print(f"Component: {len(comp_texts)}  |  Relation: {len(rel_pairs)}", flush=True)

BATCH = 64

def predict_single(model, tokenizer, texts, max_len=512):
    preds = []
    for i in range(0, len(texts), BATCH):
        enc = tokenizer(texts[i:i+BATCH], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        with torch.no_grad():
            preds.extend(torch.argmax(model(**enc).logits, dim=-1).tolist())
    return preds

def predict_pairs(model, tokenizer, pairs, max_len=256):
    preds = []
    for i in range(0, len(pairs), BATCH):
        chunk = pairs[i:i+BATCH]
        enc = tokenizer([x[0] for x in chunk], [x[1] for x in chunk],
                        truncation="only_second", max_length=max_len, padding=True, return_tensors="pt")
        with torch.no_grad():
            preds.extend(torch.argmax(model(**enc).logits, dim=-1).tolist())
    return preds

# ── 1. v7 Component (cümle only, bağlam yok) ───────────────
print("\n=== v7 modelimiz — Component (sadece cümle) ===", flush=True)
tok = AutoTokenizer.from_pretrained("models/component_classifier/final")
mod = AutoModelForSequenceClassification.from_pretrained("models/component_classifier/final")
mod.eval()
V6C = {0:"claim", 1:"evidence", 2:"other"}
MAP = {"claim":"claim", "evidence":"premise", "other":"none"}
preds = [MAP[V6C[p]] for p in predict_single(mod, tok, comp_texts)]
mf1 = f1_score(comp_true, preds, labels=["none","claim","premise"], average="macro", zero_division=0)
print(f"Macro-F1: {mf1:.4f}", flush=True)
print(classification_report(comp_true, preds, labels=["none","claim","premise"], zero_division=0), flush=True)
del mod, tok

# ── 2. v7 Relation cross-encoder ────────────────────────────
print("\n=== v7 modelimiz — Relation cross-encoder ===", flush=True)
tok = AutoTokenizer.from_pretrained("models/relation_classifier/final")
mod = AutoModelForSequenceClassification.from_pretrained("models/relation_classifier/final")
mod.eval()
V6R = {0:"support", 1:"attack", 2:"none"}
preds = [V6R[p] for p in predict_pairs(mod, tok, rel_pairs)]
mf1 = f1_score(rel_true, preds, labels=["support","attack"], average="macro", zero_division=0)
print(f"Macro-F1 (support+attack): {mf1:.4f}", flush=True)
print(classification_report(rel_true, preds, labels=["support","attack","none"], zero_division=0), flush=True)
del mod, tok
