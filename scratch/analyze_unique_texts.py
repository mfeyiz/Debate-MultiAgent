import json
from pathlib import Path

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    unique_texts = {}
    for node in nodes:
        d = node["data"]
        text = d["text"].strip()
        if text not in unique_texts:
            unique_texts[text] = {
                "kinds": set(),
                "ids": []
            }
        unique_texts[text]["kinds"].add(d["kind"])
        unique_texts[text]["ids"].append(d["id"])
        
    print(f"Total nodes: {len(nodes)}")
    print(f"Unique texts: {len(unique_texts)}")
    print("\n--- UNIQUE TEXTS AND THEIR PREDICTED KINDS ---")
    for text, info in sorted(unique_texts.items(), key=lambda x: len(x[1]["ids"]), reverse=True):
        print(f"Count: {len(info['ids'])} | Predicted Kinds: {list(info['kinds'])}")
        print(f"Text: \"{text}\"")
        print("-" * 60)

if __name__ == "__main__":
    main()
