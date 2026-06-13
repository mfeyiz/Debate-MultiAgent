import asyncio
import sys
import json
from pathlib import Path

# Add workspace directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory, init_db
from app.models import Agent, Debate
from app.services.debate_service import DebateService

async def main():
    print("Initializing DB...")
    await init_db()
    
    svc = DebateService()
    
    async with async_session_factory() as db:
        # Get Agent Alpha (Proponent) and Agent Beta (Opponent)
        proponent = await db.get(Agent, 1)
        opponent = await db.get(Agent, 2)
        
        if not proponent or not opponent:
            print("Error: Default agents not found in the database.")
            return
            
        topic = "Çocuk yapmak ahlaken doğru bi davranış mı ?"
        print(f"Creating debate: '{topic}'...")
        debate = await svc.create_debate(db, topic, [proponent.id, opponent.id], max_rounds=5)
        print(f"Debate created. ID: {debate.id}")
        
        # Start debate (Round 1, Turn 1: Proponent)
        print("Starting debate...")
        await svc.start_debate(db, debate.id)
        await db.refresh(debate)
        
        # Advance for the remaining 9 turns (Round 1 Turn 2, then Rounds 2, 3, 4, 5)
        for turn in range(2, 11):
            print(f"Advancing turn {turn}/10...")
            await svc.advance_debate(db, debate.id)
            await db.refresh(debate)
            await asyncio.sleep(0.5)
            
        print("Debate completed. Running ModernBERT analysis...")
        await svc.analyze_debate(db, debate.id)
        
        print("Retrieving argument map...")
        map_data = await svc.get_argument_map(db, debate.id)
        
        # Save to local file in workspace for analysis
        output_file = Path(__file__).resolve().parent.parent / "morality_debate_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
            
        print(f"Analysis saved to {output_file}")
        
        # Print basic transcript stats
        print(f"\n--- TRANSCRIPT ---")
        messages = sorted(list(debate.messages), key=lambda m: m.position)
        for msg in messages:
            print(f"\n[{msg.agent.name} ({msg.message_type})]:")
            print(msg.current_version.content)
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
