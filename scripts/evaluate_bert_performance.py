import json
from pathlib import Path

def get_ground_truth_kind(text):
    text_lower = text.lower().strip()
    
    # 1. Other (Greetings, transition phrases, and sentence fragments)
    if "elbette, karşıt browser" in text_lower:
        if len(text_lower) < 60:
            return "other"
    if "elbette, karşıt browser’ın" in text_lower:
        if len(text_lower) < 60:
            return "other"
    if "evrensel beyannamesi’nin 16." in text_lower or "bildirgesi’nin 16." in text_lower or "bildirgesi'nin 16." in text_lower:
        if text_lower.endswith("16.") or text_lower.endswith("16"):
            return "other"
            
    # 2. Claims (Main thesis statements)
    if text_lower == "çocuk yapmak ahlaken doğru bir davranıştır" or text_lower == "çocuk yapmak ahlaken doğru bir davranıştır;" or text_lower == "çocuk yapmanın ahlaken doğru olduğunu savunmaya devam ediyorum;":
        return "claim"
    if "çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez" in text_lower:
        return "claim"
        
    # 3. Evidence (All supporting arguments, reasoning, and citations)
    return "evidence"

def get_ground_truth_relation(src_text, src_agent, src_stance, tgt_text, tgt_agent, tgt_stance):
    src_lower = src_text.lower().strip()
    tgt_lower = tgt_text.lower().strip()
    
    # Identify if targets are main claims
    tgt_is_proponent_claim = (
        tgt_lower == "çocuk yapmak ahlaken doğru bir davranıştır" or 
        tgt_lower == "çocuk yapmak ahlaken doğru bir davranıştır;" or 
        tgt_lower == "çocuk yapmanın ahlaken doğru olduğunu savunmaya devam ediyorum;"
    )
    tgt_is_opponent_claim = (
        "çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez" in tgt_lower
    )
    
    # 1. Relation to main claims
    if tgt_is_proponent_claim:
        # If opponent source is attacking the proponent's claim
        if "karşıt" in src_agent.lower():
            return "attack"
        else:
            return "support"
            
    if tgt_is_opponent_claim:
        # If proponent source is attacking the opponent's claim
        if "savunucu" in src_agent.lower():
            return "attack"
        else:
            return "support"
            
    # 2. Relation to human rights article fragments
    # If the target is the fragment "BM İnsan Hakları...", check the source.
    tgt_is_fragment = (
        "evrensel beyannamesi’nin 16." in tgt_lower or 
        "bildirgesi’nin 16." in tgt_lower or 
        "bildirgesi'nin 16." in tgt_lower
    )
    if tgt_is_fragment:
        # Only the actual article text supports/completes it
        if "maddesi" in src_lower or "bu hak" in src_lower:
            return "support"
        # Other unrelated arguments (like ecology, climate, Benatar's asymmetry) have NO relation to the human rights fragment
        if "iklim" in src_lower or "ekolojik" in src_lower or "benatar" in src_lower or "asimetri" in src_lower:
            return "none"
        # If it's by the same speaker talking about the rights, it's support
        if "savunucu" in src_agent.lower() and "karşıt" not in src_agent.lower():
            return "support"
        return "none"

    # 3. Specific attacks between opposing sides
    # If Proponent refutes Benatar's asymmetry arg (which is Opponent's argument)
    if "benatar" in src_lower and "asimetri" in src_lower and "savunucu" in src_agent.lower():
        if "benatar" in tgt_lower or "asimetri" in tgt_lower:
            return "attack"
            
    # If Opponent refutes proponent's argument about parents' responsibility / logic of consent
    if "rıza sorununun" in src_lower and "yanıltıcıdır" in src_lower and "karşıt" in src_agent.lower():
        if "rıza sorunu" in tgt_lower:
            return "attack"

    # If same agent discussing unrelated things, no relation
    if "iklim" in src_lower and "rıza" in tgt_lower:
        return "none"
    if "rıza" in src_lower and "iklim" in tgt_lower:
        return "none"
        
    # Default fallback
    if "savunucu" in src_agent.lower() and "savunucu" in tgt_agent.lower():
        return "support"
    if "karşıt" in src_agent.lower() and "karşıt" in tgt_agent.lower():
        return "support"
    if ("savunucu" in src_agent.lower() and "karşıt" in tgt_agent.lower()) or ("karşıt" in src_agent.lower() and "savunucu" in tgt_agent.lower()):
        return "attack"
        
    return "none"

def calculate_metrics(y_true, y_pred, labels):
    metrics = {}
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0
    error_rate = 1.0 - accuracy
    
    metrics["accuracy"] = accuracy
    metrics["error_rate"] = error_rate
    metrics["class_metrics"] = {}
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics["class_metrics"][label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": y_true.count(label)
        }
        
    return metrics

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    # 1. Evaluate Components
    node_true = []
    node_pred = []
    component_details = []
    
    for node in nodes:
        d = node["data"]
        text = d["text"].strip()
        pred_kind = d["kind"].lower()
        gt_kind = get_ground_truth_kind(text)
        
        node_true.append(gt_kind)
        node_pred.append(pred_kind)
        
        component_details.append({
            "id": d["id"],
            "speaker": d["agentName"],
            "text": text,
            "pred": pred_kind,
            "gt": gt_kind,
            "correct": pred_kind == gt_kind
        })
        
    comp_metrics = calculate_metrics(node_true, node_pred, ["claim", "evidence", "other"])
    
    # 2. Evaluate Relations
    node_map = {n["data"]["id"]: n["data"] for n in nodes}
    edge_true = []
    edge_pred = []
    relation_details = []
    
    # We evaluate existing edges in the analysis
    for edge in edges:
        d = edge["data"]
        src = node_map.get(d["source"])
        tgt = node_map.get(d["target"])
        if not src or not tgt:
            continue
            
        pred_rel = d["relationType"].lower()
        gt_rel = get_ground_truth_relation(
            src["text"], src["agentName"], src["stance"],
            tgt["text"], tgt["agentName"], tgt["stance"]
        )
        
        edge_true.append(gt_rel)
        edge_pred.append(pred_rel)
        
        relation_details.append({
            "id": d["id"],
            "src_text": src["text"],
            "tgt_text": tgt["text"],
            "src_speaker": src["agentName"],
            "tgt_speaker": tgt["agentName"],
            "pred": pred_rel,
            "gt": gt_rel,
            "correct": pred_rel == gt_rel
        })
        
    rel_metrics = calculate_metrics(edge_true, edge_pred, ["support", "attack", "none"])
    
    # 3. Write Report to morality_evaluation_results.md
    report_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_evaluation_results.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ModernBERT Argument Extraction Evaluation Report\n\n")
        f.write("This report evaluates the performance of the fine-tuned ModernBERT models in extracting argument structures (components) and relationships (relations) from a 5-round debate on: **\"Çocuk yapmak ahlaken doğru bi davranış mı ?\"**.\n\n")
        
        f.write("## 1. Overall Performance Metrics\n\n")
        f.write("### Component Classification\n")
        f.write(f"- **Accuracy**: {comp_metrics['accuracy']:.2%}\n")
        f.write(f"- **Error Rate**: {comp_metrics['error_rate']:.2%}\n\n")
        
        f.write("| Component Type | Count | Precision | Recall | F1-Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for lbl, m in comp_metrics["class_metrics"].items():
            f.write(f"| {lbl.upper()} | {m['count']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |\n")
            
        f.write("\n### Relation Classification\n")
        f.write(f"- **Accuracy**: {rel_metrics['accuracy']:.2%}\n")
        f.write(f"- **Error Rate**: {rel_metrics['error_rate']:.2%}\n\n")
        
        f.write("| Relation Type | Count | Precision | Recall | F1-Score |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for lbl, m in rel_metrics["class_metrics"].items():
            f.write(f"| {lbl.upper()} | {m['count']} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} |\n")
            
        f.write("\n## 2. Component Classification Details\n\n")
        f.write("| ID | Speaker | Component Text | Predicted | Ground Truth | Correct? |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for c in component_details:
            correct_str = "✅" if c["correct"] else "❌"
            f.write(f"| {c['id']} | {c['speaker']} | {c['text']} | {c['pred'].upper()} | {c['gt'].upper()} | {correct_str} |\n")
            
        f.write("\n## 3. Relation Classification Details\n\n")
        f.write("| ID | Source Text | Target Text | Predicted | Ground Truth | Correct? |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for r in relation_details:
            correct_str = "✅" if r["correct"] else "❌"
            f.write(f"| {r['id']} | {r['src_text']} | {r['tgt_text']} | {r['pred'].upper()} | {r['gt'].upper()} | {correct_str} |\n")
            
    print(f"Evaluation report generated successfully at: {report_path}")
    print(f"Component Classification - Accuracy: {comp_metrics['accuracy']:.2%}, Error Rate: {comp_metrics['error_rate']:.2%}")
    print(f"Relation Classification  - Accuracy: {rel_metrics['accuracy']:.2%}, Error Rate: {rel_metrics['error_rate']:.2%}")

if __name__ == "__main__":
    main()
