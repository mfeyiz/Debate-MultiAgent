import asyncio
import sys
from pathlib import Path

# Add workspace directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import async_session_factory, init_db
from app.services.debate_service import DebateService

async def main():
    print("Initializing DB...")
    await init_db()
    
    svc = DebateService()
    
    async with async_session_factory() as db:
        debate_id = 14
        print(f"Running analyze_debate on debate {debate_id}...")
        try:
            res = await svc.analyze_debate(db, debate_id)
            print("Analysis succeeded!")
            print(res)
        except Exception as e:
            import traceback
            print("Analysis failed with error:")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
