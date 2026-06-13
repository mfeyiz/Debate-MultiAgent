import asyncio
from sqlalchemy import select
from app.database import async_session_factory, init_db
from app.models import Agent, Debate

async def main():
    await init_db()
    async with async_session_factory() as db:
        agents = (await db.execute(select(Agent))).scalars().all()
        print("AGENTS:")
        for a in agents:
            print(f"- ID: {a.id}, Name: {a.name}, Role: {a.role}, Model: {a.model_name}")
        
        debates = (await db.execute(select(Debate))).scalars().all()
        print("\nDEBATES:")
        for d in debates:
            print(f"- ID: {d.id}, Topic: {d.topic}, Status: {d.status}, Rounds: {d.current_round}/{d.max_rounds}")

if __name__ == "__main__":
    asyncio.run(main())
