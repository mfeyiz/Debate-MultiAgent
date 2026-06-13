import asyncio
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory, init_db
from app.services.debate_service import DebateService

async def main():
    await init_db()
    svc = DebateService()
    
    async with async_session_factory() as db:
        print("Retrieving argument map for debate ID 12...")
        map_data = await svc.get_argument_map(db, 12)
        
        print("\n==================================================")
        print("MODERNBERT ANALYSIS RESULTS (RUN 17)")
        print("==================================================")
        
        print("\n--- Extracted Argument Components (Nodes) ---")
        nodes = map_data.get("nodes", [])
        print(f"Total Components Found: {len(nodes)}")
        for node in nodes:
            data = node["data"]
            print(f"Component ID: {data['id']}")
            print(f"  Speaker: {data['agentName']}")
            print(f"  Round: {data['round']}")
            print(f"  Type: {data['labelType']} ({data['kind']})")
            print(f"  Text: \"{data['text']}\"")
            print(f"  Confidence: {data['confidence']:.3f}")
            print(f"  Relation to Topic ({data['topicRelationType']}): Conf={data['topicRelationConfidence']:.3f}")
            print("-" * 40)
            
        print("\n--- Classified Relations (Edges) ---")
        edges = map_data.get("edges", [])
        print(f"Total Graph Edges Found: {len(edges)}")
        component_by_id = {node["data"]["id"]: node["data"] for node in nodes}
        for edge in edges:
            data = edge["data"]
            source_node = component_by_id.get(data['source'])
            target_node = component_by_id.get(data['target'])
            
            source_txt = source_node['text'] if source_node else "Unknown"
            target_txt = target_node['text'] if target_node else "Unknown"
            source_spk = source_node['agentName'] if source_node else "Unknown"
            target_spk = target_node['agentName'] if target_node else "Unknown"
            
            print(f"Relation ID: {data['id']}")
            print(f"  Type: {data['relationType'].upper()}")
            print(f"  Confidence: {data['confidence']:.3f}")
            print(f"  Source Component: [{source_spk}] \"{source_txt}\"")
            print(f"  Target Component: [{target_spk}] \"{target_txt}\"")
            print(f"  Distance: {data['distanceTurns']} turns")
            print(f"  Is Long Range: {data['isLongRange']}")
            print("-" * 40)
            
        print("\n--- Key Findings ---")
        findings = map_data.get("findings", [])
        for idx, finding in enumerate(findings, 1):
            print(f"{idx}. [{finding['type'].upper()}] - {finding['title']}")
            print(f"   Detail: {finding.get('detail', 'N/A')}")
            print(f"   Confidence: {finding.get('confidence', 0.0):.3f}")
            
        artifact_dir = Path("/Users/mfeyiz/.gemini/antigravity/brain/cdf9b05e-f522-44ed-9e0c-a3c22077e83c")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        export_path = artifact_dir / "debate_argument_map.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        print(f"\nArgument map exported to: {export_path}")
        print("==================================================")

if __name__ == "__main__":
    asyncio.run(main())
