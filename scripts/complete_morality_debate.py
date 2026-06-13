import asyncio
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add workspace directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory, init_db
from app.models import Agent, Debate, Message, MessageVersion, DebateParticipant
from app.services.debate_service import DebateService
from sqlalchemy import select

async def main():
    print("Initializing DB...")
    await init_db()
    
    svc = DebateService()
    
    async with async_session_factory() as db:
        debate_id = 14
        debate = await db.get(Debate, debate_id)
        if not debate:
            print(f"Error: Debate {debate_id} not found.")
            return
            
        messages = list(debate.messages)
        print(f"Current message count: {len(messages)}")
        
        if len(messages) < 10:
            # We need to generate the 10th message (position 9)
            # Position 8 was posted by Agent 4. Next is Agent 3 (Opponent).
            stmt = select(DebateParticipant).filter_by(debate_id=debate_id).order_by(DebateParticipant.position.asc())
            res = await db.execute(stmt)
            participants = res.scalars().all()
            
            # Agent 3 is the opponent
            opponent_participant = next(p for p in participants if p.role_in_debate == "opponent")
            opponent = opponent_participant.agent
            
            print(f"Generating 10th response for opponent: {opponent.name}...")
            
            # Generate response
            response_text = await svc.agent_svc.generate_response(
                agent_config=opponent,
                topic=debate.topic,
                messages=messages,
            )
            
            print(f"Generated response text:\n{response_text}")
            
            # Insert message
            last_msg = messages[-1]
            msg = Message(
                debate_id=debate_id,
                agent_id=opponent.id,
                parent_id=last_msg.id,
                message_type="attack",
                position=9,
            )
            db.add(msg)
            await db.flush()
            
            version = MessageVersion(
                message_id=msg.id,
                version_number=1,
                content=response_text,
                is_current=True,
            )
            db.add(version)
            await db.commit()
            
            print("10th message saved to database.")
            
        # Refresh debate
        await db.refresh(debate)
        print(f"Total messages now: {len(debate.messages)}")
        
        # Set status to resolved
        debate.status = "resolved"
        debate.current_round = 5
        await db.commit()
        
        # Run ModernBERT analysis
        print("Running full ModernBERT analysis...")
        await svc.analyze_debate(db, debate_id)
        print("Analysis completed successfully.")
        
        # Fetch argument map
        print("Fetching argument map...")
        map_data = await svc.get_argument_map(db, debate_id)
        
        output_file = Path(__file__).resolve().parent.parent / "morality_debate_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(map_data, f, ensure_ascii=False, indent=2)
            
        print(f"Argument map saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
