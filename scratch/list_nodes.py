import json
from pathlib import Path

def main():
    json_path = Path("/Users/mfeyiz/Desktop/Debate-MultiAgent/morality_debate_analysis.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    nodes = data.get("nodes", [])
    print(f"Total nodes: {len(nodes)}")
    for i, node in enumerate(nodes):
        d = node["data"]
        print(f"{i+1}. ID: {d['id']} | Speaker: {d['agentName']} | Kind: {d['kind']}")
        print(f"   Text: {d['text']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
