import json

export_path = "/Users/mfeyiz/.gemini/antigravity/brain/cdf9b05e-f522-44ed-9e0c-a3c22077e83c/debate_argument_map.json"
with open(export_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
edges = data.get("edges", [])
component_by_id = {node["data"]["id"]: node["data"] for node in nodes}

attacks = [e["data"] for e in edges if e["data"]["relationType"] == "attack"]
print(f"Total attacks: {len(attacks)}")
for a in attacks:
    src = component_by_id.get(a['source'], {}).get('text', 'UNKNOWN')
    src_spk = component_by_id.get(a['source'], {}).get('agentName', 'UNKNOWN')
    tgt = component_by_id.get(a['target'], {}).get('text', 'UNKNOWN')
    tgt_spk = component_by_id.get(a['target'], {}).get('agentName', 'UNKNOWN')
    print(f"- ATTACK (Conf={a['confidence']:.3f}): [{src_spk}] \"{src}\" -> [{tgt_spk}] \"{tgt}\"")
    print("-" * 50)
