import json
from pathlib import Path

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    node_map = {n["data"]["id"]: n["data"] for n in nodes}
    
    # Let's inspect the edges
    print(f"Total edges: {len(edges)}")
    for i, edge in enumerate(edges):
        d = edge["data"]
        src = node_map.get(d["source"])
        tgt = node_map.get(d["target"])
        if not src or not tgt:
            continue
        print(f"Index: {i} | EdgeID: {d['id']} | Pred: {d['relationType'].upper()}")
        print(f"  Src: \"{src['text'][:60]}...\" (Agent: {src['agentName']}, Stance: {src['stance']})")
        print(f"  Tgt: \"{tgt['text'][:60]}...\" (Agent: {tgt['agentName']}, Stance: {tgt['stance']})")
        print("-" * 50)

if __name__ == "__main__":
    main()
