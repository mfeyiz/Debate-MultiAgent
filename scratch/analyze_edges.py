import json
from pathlib import Path

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    node_map = {n["data"]["id"]: n["data"] for n in nodes}
    
    print(f"Total edges: {len(edges)}")
    print("="*100)
    for i, edge in enumerate(edges):
        d = edge["data"]
        src = node_map.get(d["source"])
        tgt = node_map.get(d["target"])
        if not src or not tgt:
            continue
            
        print(f"{i+1}. Edge ID: {d['id']} | Predicted: {d['relationType'].upper()} (Conf: {d['confidence']:.3f})")
        print(f"   Source: [{src['agentName']}] (Stance: {src['stance']}): \"{src['text']}\"")
        print(f"   Target: [{tgt['agentName']}] (Stance: {tgt['stance']}): \"{tgt['text']}\"")
        print("-" * 100)

if __name__ == "__main__":
    main()
