import json
from pathlib import Path

def get_ground_truth_kind(text):
    text_lower = text.lower().strip()
    
    # Other / Meta-talk / Fragments
    if "elbette, karşıt browser" in text_lower:
        return "other"
    if "elbette, karşıt browser’ın" in text_lower:
        return "other"
    if "evrensel beyannamesi’nin 16." in text_lower or "bildirgesi’nin 16." in text_lower or "bildirgesi'nin 16." in text_lower:
        # Fragments of BM Human Rights Declaration (split from "maddesi ...")
        if text_lower.endswith("16.") or text_lower.endswith("16"):
            return "other"
            
    # Main Claims
    # Proponent main thesis
    if text_lower == "çocuk yapmak ahlaken doğru bir davranıştır" or text_lower == "çocuk yapmak ahlaken doğru bir davranıştır;" or text_lower == "çocuk yapmanın ahlaken doğru olduğunu savunmaya devam ediyorum;":
        return "claim"
    # Opponent main thesis
    if "çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez" in text_lower:
        return "claim"
        
    # Evidence / Premises (default for arguments in the debate)
    return "evidence"

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    
    print("Unique component texts and their mapped ground truth kind:")
    print("="*80)
    unique_texts = {}
    for node in nodes:
        d = node["data"]
        text = d["text"].strip()
        if text not in unique_texts:
            unique_texts[text] = {
                "pred": d["kind"],
                "gt": get_ground_truth_kind(text)
            }
            
    for text, info in sorted(unique_texts.items(), key=lambda x: x[1]["gt"]):
        print(f"GT: {info['gt'].upper()} | PRED: {info['pred'].upper()} | Text: \"{text}\"")
        print("-" * 80)

if __name__ == "__main__":
    main()
