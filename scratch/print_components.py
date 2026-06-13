import json

export_path = "/Users/mfeyiz/.gemini/antigravity/brain/cdf9b05e-f522-44ed-9e0c-a3c22077e83c/debate_argument_map.json"
with open(export_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
edges = data.get("edges", [])

print(f"Total nodes: {len(nodes)}")
print("NODES:")
for n in nodes:
    d = n["data"]
    print(f"- {d['id']} [{d['agentName']}] ({d['kind']}): \"{d['text']}\"")

print(f"\nTotal edges: {len(edges)}")
print("EDGES:")
component_by_id = {node["data"]["id"]: node["data"] for node in nodes}
for e in edges:
    d = e["data"]
    src = component_by_id.get(d['source'], {}).get('text', 'UNKNOWN')
    src_spk = component_by_id.get(d['source'], {}).get('agentName', 'UNKNOWN')
    tgt = component_by_id.get(d['target'], {}).get('text', 'UNKNOWN')
    tgt_spk = component_by_id.get(d['target'], {}).get('agentName', 'UNKNOWN')
    print(f"- {d['relationType'].upper()} (Conf={d['confidence']:.3f}): [{src_spk}] \"{src}\" -> [{tgt_spk}] \"{tgt}\"")
