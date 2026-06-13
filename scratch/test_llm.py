import asyncio
from app.config import Config
from app.services.agent_service import AgentService
from app.models import Agent as AgentModel

async def test_llm():
    print(f"API Key present: {bool(Config.OPENROUTER_API_KEY)}")
    print(f"Default Model: {Config.DEFAULT_MODEL}")
    agent_config = AgentModel(
        name="Test",
        role="proponent",
        system_prompt="You are a helper.",
        temperature=0.7
    )
    svc = AgentService()
    print(f"Mock mode: {svc._mock_mode}")
    try:
        response = await svc.generate_opening_claim(agent_config, "Yapay zeka")
        print("\nSUCCESS!")
        print(f"Response: {response}")
    except Exception as e:
        print("\nERROR:")
        print(e)

if __name__ == "__main__":
    asyncio.run(test_llm())
