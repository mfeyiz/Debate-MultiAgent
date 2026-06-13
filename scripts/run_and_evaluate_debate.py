import asyncio
import sys
import os
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory, init_db
from app.models import Agent, Debate, Message
from app.services.debate_service import DebateService

async def run_debate_and_evaluate():
    print("==================================================")
    print("Initializing Database...")
    await init_db()
    
    svc = DebateService()
    
    async with async_session_factory() as db:
        print("Checking default agents...")
        # Get default agents: ID 1 is Alpha (Proponent), ID 2 is Beta (Opponent)
        proponent = await db.get(Agent, 1)
        opponent = await db.get(Agent, 2)
        
        if not proponent or not opponent:
            print("Error: Seed agents not found in the database.")
            return
            
        print(f"Proponent Agent: {proponent.name} ({proponent.model_name})")
        print(f"Opponent Agent: {opponent.name} ({opponent.model_name})")
        
        topic = "Çocuk yapmak ahlaken doğru bir davranış mı?"
        print(f"\nCreating new debate with topic: '{topic}'...")
        debate = await svc.create_debate(db, topic, [proponent.id, opponent.id], max_rounds=5)
        print(f"Debate created. ID: {debate.id}, Status: {debate.status}")
        
        # Start the debate
        print("\nStarting debate (generating turn 1 - Proponent opening claim)...")
        await svc.start_debate(db, debate.id)
        
        # Refresh debate to get the opening message
        await db.refresh(debate)
        messages = list(debate.messages)
        opening_msg = messages[-1]
        print(f"\n[Turn 1] Proponent (Agent Alpha):")
        print(opening_msg.current_version.content)
        
        # Run the next 9 turns (to complete 5 rounds - total 10 messages)
        for turn in range(2, 11):
            speaker_name = "Opponent (Agent Beta)" if turn % 2 == 0 else "Proponent (Agent Alpha)"
            print(f"\nAdvancing debate to turn {turn} ({speaker_name})...")
            
            # We track if the message is regenerated
            current_count = len(debate.messages)
            await svc.advance_debate(db, debate.id)
            await db.refresh(debate)
            
            new_messages = list(debate.messages)
            if len(new_messages) > current_count:
                latest_msg = new_messages[-1]
                print(f"[Turn {turn}] {speaker_name}:")
                # Print all versions to show if it went through the regeneration loop
                stmt_versions = latest_msg.versions.all()
                for version in stmt_versions:
                    tag = " (Current)" if version.is_current else ""
                    print(f"--- Version {version.version_number}{tag} ---")
                    print(version.content)
                    if version.strength_score is not None:
                        print(f"ModernBERT Strength Score: {version.strength_score}")
                    # If there's feedback, print it
                    analyses = version.analyses.all()
                    if analyses:
                        for analysis in analyses:
                            print(f"Feedback from BERT: {analysis.feedback_text}")
            else:
                print("Failed to advance debate.")
                break
                
            await asyncio.sleep(0.5) # small delay
            
        print("\n==================================================")
        print("Debate completed! Refreshing debate data...")
        await db.refresh(debate)
        print(f"Final Debate Status: {debate.status}, Current Round: {debate.current_round}")
        
        print("\nRunning full ModernBERT analysis on the debate...")
        # analyze_debate runs the component and relation classification over the entire transcript
        analysis_map = await svc.analyze_debate(db, debate.id)
        print("Full ModernBERT analysis completed successfully.")
        
        print("\nRetrieving argument map data...")
        # Retrieve the latest completed analysis run to inspect nodes and edges
        map_data = await svc.get_argument_map(db, debate.id)
        
        # Print out the results in a structured way for the evaluation report
        print("\n==================================================")
        print("MODERNBERT ANALYSIS RESULTS")
        print("==================================================")
        
        print("\n--- Extracted Argument Components (Nodes) ---")
        nodes = map_data.get("nodes", [])
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
            
        # Also print findings and impact
        print("\n--- Key Findings ---")
        findings = map_data.get("findings", [])
        for idx, finding in enumerate(findings, 1):
            print(f"{idx}. [{finding['type'].upper()}] - {finding['title']}")
            print(f"   Summary: {finding['summary']}")
            print(f"   Metric: {finding['metric_value']}")
            
        # Export the map to a JSON file in the artifacts directory for future reference
        artifact_dir = Path("/Users/mfeyiz/.gemini/antigravity/brain/cdf9b05e-f522-44ed-9e0c-a3c22077e83c")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        export_path = artifact_dir / "debate_argument_map.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
        print(f"\nArgument map exported to: {export_path}")
        print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_debate_and_evaluate())
