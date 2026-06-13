import json
from pathlib import Path

def main():
    json_path = Path(__file__).resolve().parent.parent / "morality_debate_analysis.json"
    if not json_path.exists():
        print("Error: Analysis file not found.")
        return
        
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    print("==================================================")
    print(f"EXTRACTED COMPONENTS (TOTAL: {len(nodes)})")
    print("==================================================")
    for node in nodes:
        d = node["data"]
        print(f"ID: {d['id']} | Round: {d['round']} | Speaker: {d['agentName'][:20]}...")
        print(f"  Text: \"{d['text']}\"")
        print(f"  BERT Type: {d['kind'].upper()} (Conf: {d['confidence']:.3f})")
        print("-" * 50)
        
    print("\n==================================================")
    print(f"EXTRACTED RELATIONS (TOTAL: {len(edges)})")
    print("==================================================")
    component_by_id = {node["data"]["id"]: node["data"] for node in nodes}
    for edge in edges:
        d = edge["data"]
        src = component_by_id.get(d['source'], {})
        tgt = component_by_id.get(d['target'], {})
        print(f"ID: {d['id']} | relationType: {d['relationType'].upper()} (Conf: {d['confidence']:.3f})")
        print(f"  Source: \"{src.get('text', 'Unknown')}\"")
        print(f"  Target: \"{tgt.get('text', 'Unknown')}\"")
        print("-" * 50)

if __name__ == "__main__":
    main()
