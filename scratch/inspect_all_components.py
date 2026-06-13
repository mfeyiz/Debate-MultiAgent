import json

export_path = "/Users/mfeyiz/.gemini/antigravity/brain/cdf9b05e-f522-44ed-9e0c-a3c22077e83c/debate_argument_map.json"
with open(export_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nodes = data.get("nodes", [])
print(f"Total nodes: {len(nodes)}")
for n in nodes:
    d = n["data"]
    print(f"- Node ID: {d['id']} | Speaker: {d['agentName']} | Round: {d['round']} | Type: {d['kind'].upper()} (Conf={d['confidence']:.3f})")
    print(f"  Text: \"{d['text']}\"")
    print(f"  Topic relation: {d['topicRelationType']} (Conf={d['topicRelationConfidence']:.3f})")
    print("-" * 50)
