"""LLM Agent service powered by Pydantic-AI and OpenRouter."""

from __future__ import annotations

import os
import random
from typing import Dict, List

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import Config
from app.models import Agent as AgentModel
from app.models import Message, MessageVersion


class AgentService:
    """Manages LLM agents for the debate platform."""

    def __init__(self) -> None:
        self._mock_mode = not Config.OPENROUTER_API_KEY or Config.OPENROUTER_API_KEY.startswith("your-")
        if not self._mock_mode:
            provider = OpenAIProvider(
                base_url=Config.OPENROUTER_BASE_URL,
                api_key=Config.OPENROUTER_API_KEY,
            )
            self._model = OpenAIChatModel(
                Config.DEFAULT_MODEL,
                provider=provider,
            )
        else:
            self._model = None

    def _build_system_prompt(self, agent_config: AgentModel, topic: str) -> str:
        """Compose the system prompt for an agent."""
        base = agent_config.system_prompt or self._default_system_prompt(agent_config.role)
        return (
            f"{base}\n\n"
            f"Yapılandırılmış bir tartışmaya katılıyorsunuz.\n"
            f"Tartışma konusu: {topic}\n"
            f"Atanan rolünüz: {agent_config.role.upper()}.\n"
            f"Türkçe olarak, açık ve öz bir paragraf halinde yanıt verin. Mümkünse kaynak belirtin.\n"
        )

    @staticmethod
    def _default_system_prompt(role: str) -> str:
        prompts = {
            "proponent": (
                "Kesin bir savunucusunuz. Göreviniz, kanıt, veri ve mantıksal akıl yürütme kullanarak "
                "iyi desteklenmiş iddialar oluşturmaktır. Argümanlarınızı her zaman somut gerçeklere dayandırın."
            ),
            "opponent": (
                "Eleştirel bir karşıtsınız. Göreviniz, karşı görüşteki zayıflıkları, çelişkileri ve "
                "mantıksal boşlukları tespit etmektir. İddiaları hassasiyetle çürütün ve mümkünse karşı kanıtlara atıfta bulunun."
            ),
            "moderator": (
                "Tarafsız bir moderatörsünüz. Argümanları sentezleyin, mantıksal hataları vurgulayın ve "
                "tartışmayı odaklı tutun. Yeni birincil argümanlar sunmayın; yalnızca değerlendirin ve özetleyin."
            ),
        }
        return prompts.get(role, prompts["proponent"])

    def _build_context_prompt(
        self,
        messages: List[Message],
        agent_config: AgentModel,
        feedback: str | None = None,
    ) -> str:
        """Build the user prompt from debate history and optional feedback."""
        lines: List[str] = []
        lines.append("### Tartışma Geçmişi ###\n")
        for msg in messages:
            agent_name = msg.agent.name if msg.agent else "Sistem"
            cv = msg.current_version
            content = cv.content if cv else ""
            lines.append(f"{agent_name} ({msg.message_type}): {content}\n")

        if feedback:
            lines.append(
                "\n### MODERATÖR GERİ BİLDİRİMİ ###\n"
                f"{feedback}\n\n"
                "Sonraki yanıtınızı formüle ederken yukarıdaki geri bildirimi dikkate alın.\n"
            )

        lines.append(f"\nŞimdi sıra sizde, {agent_config.name}. Doğrudan yanıt verin:")
        return "\n".join(lines)

    async def generate_response(
        self,
        agent_config: AgentModel,
        topic: str,
        messages: List[Message],
        feedback: str | None = None,
    ) -> str:
        """Run the agent and return its text response."""
        if self._mock_mode:
            return self._mock_response(agent_config, topic, messages, feedback)

        system_prompt = self._build_system_prompt(agent_config, topic)
        user_prompt = self._build_context_prompt(
            messages, agent_config, feedback=feedback
        )

        agent = Agent(
            self._model,
            system_prompt=system_prompt,
            model_settings={"temperature": agent_config.temperature},
        )
        result = await agent.run(user_prompt)
        return str(result.output)

    async def generate_opening_claim(
        self, agent_config: AgentModel, topic: str
    ) -> str:
        """Generate the opening claim for a debate."""
        if self._mock_mode:
            return self._mock_opening_claim(agent_config, topic)

        system_prompt = self._build_system_prompt(agent_config, topic)
        user_prompt = (
            f"Tartışma konusu: {topic}\n\n"
            "Açılış savunucusu olarak, temel iddianızı açık ve öz bir şekilde belirtin. "
            "Güçlü bir destekleyici kanıt veya gerekçe sunun.\n"
            "Türkçe olarak tek bir iyi yapılandırılmış paragraf halinde yanıt verin."
        )
        agent = Agent(
            self._model,
            system_prompt=system_prompt,
            model_settings={"temperature": agent_config.temperature},
        )
        result = await agent.run(user_prompt)
        return str(result.output)

    # ------------------------------------------------------------------
    # Mock fallback for demo without API keys
    # Uses topic-aware templates instead of hardcoded unrelated content.
    # ------------------------------------------------------------------

    _MOCK_TOPIC_KEYWORDS: Dict[str, List[str]] = {
        "default": ["bu konu", "tartışma konusu", "iddia"],
        "technology": ["yapay zeka", "teknoloji", "dijital dönüşüm"],
        "climate": ["iklim değişikliği", "çevre", "sürdürülebilirlik"],
        "economy": ["ekonomi", "finans", "piyasa"],
        "health": ["sağlık", "tıp", "hastalık"],
        "education": ["eğitim", "öğrenme", "akademi"],
        "politics": ["siyaset", "hükümet", "politika"],
    }

    @staticmethod
    def _detect_topic_category(topic: str) -> str:
        """Detect broad topic category for generating relevant mock responses."""
        topic_lower = topic.lower()
        tech_keywords = ["yapay zeka", "ai", "teknoloji", "yazılım", "digital", "robot", "otomasyon"]
        climate_keywords = ["iklim", "çevre", "küresel", "karbon", "yeşil", "sürdürülebilir"]
        economy_keywords = ["ekonomi", "finans", "borsa", "enflasyon", "dolar", "yatırım"]
        health_keywords = ["sağlık", "tıp", "hastalık", "virüs", "aşı", "tedavi"]
        education_keywords = ["eğitim", "üniversite", "öğrenci", "öğretmen", "akademi", "okul"]
        politics_keywords = ["siyaset", "hükümet", "seçim", "parti", "devlet", "politika"]

        for category, keywords in [
            ("technology", tech_keywords),
            ("climate", climate_keywords),
            ("economy", economy_keywords),
            ("health", health_keywords),
            ("education", education_keywords),
            ("politics", politics_keywords),
        ]:
            if any(kw in topic_lower for kw in keywords):
                return category
        return "default"

    def _generate_topic_aware_mock(
        self,
        topic: str,
        role: str,
        message_type: str,
        feedback: str | None = None,
    ) -> str:
        """Generate topic-aware mock responses that reference the actual debate topic."""
        category = self._detect_topic_category(topic)
        keywords = self._MOCK_TOPIC_KEYWORDS.get(category, self._MOCK_TOPIC_KEYWORDS["default"])
        keyword = random.choice(keywords)

        templates = {
            "proponent_claim": [
                f"{topic} konusunda kesin bir şekilde savunulmalıdır. {keyword} üzerine yapılan araştırmalar, bu yöndeki iddiaları güçlü şekilde desteklemektedir.",
                f"Tartışma konusu olan '{topic}' önemli bir meseledir. {keyword} bağlamında ele alındığında, savunulacak güçlü argümanlar mevcuttur.",
                f"{topic} konusundaki iddiamızı {keyword} perspektifinden destekleyerek açıklayalım. Somut veriler ve mantıksal akıl yürütme ile bu savı temellendirebiliriz.",
            ],
            "opponent_attack": [
                f"Karşı tarafın '{topic}' konusundaki iddiası zayıf temellere dayanıyor. {keyword} ile ilgili sunulan kanıtlar yetersiz ve çelişkili görünmektedir.",
                f"{topic} konusundaki savın temelindeki varsayımları sorgulamak gerekir. {keyword} açısından bakıldığında, önemli mantıksal boşluklar mevcuttur.",
                f"Karşı görüşteki '{topic}' iddiası, {keyword} bağlamında ele alındığında tutarsızlıklar içermektedir. Bu zayıflıkları gözler önüne sermek gerekiyor.",
            ],
            "proponent_rebuttal": [
                f"Karşı tarafın eleştirilerine rağmen, '{topic}' konusundaki savımızı {keyword} verileriyle güçlendirebiliriz. Eleştirilerin temelindeki yanlış anlamaları düzeltelim.",
                f"{topic} tartışmasında karşı argümanlar göz ardı edilemez, ancak {keyword} perspektifinden bakıldığında bizim savımız daha tutarlı durmaktadır.",
                f"Karşı tarafın '{topic}' konusundaki itirazları, {keyword} alanındaki gelişmeler ışığında geçersiz kalmaktadır. Güncel bulgular bizim lehimize işlemektedir.",
            ],
            "moderator_summary": [
                f"Tartışma konusu '{topic}' bağlamında her iki tarafın da {keyword} ile ilgili argümanları dikkate alınmalıdır. Tarafların güçlü ve zayıf yönlerini özetleyelim.",
                f"{topic} meselesinde, {keyword} perspektifinden bakıldığında her iki görüşün de geçerli noktaları vardır. Mantıksal tutarlılık açısından değerlendirme yapalım.",
            ],
        }

        # Select template based on role and message type
        if role == "opponent" or message_type in ("attack",):
            key = "opponent_attack"
        elif message_type in ("rebuttal",) or role == "proponent":
            key = "proponent_rebuttal"
        elif role == "moderator":
            key = "moderator_summary"
        else:
            key = "proponent_claim"

        response = random.choice(templates.get(key, templates["proponent_claim"]))

        # Append feedback hint if regeneration feedback exists
        if feedback:
            response += f" (Geri bildirime göre düzenlendi: {feedback[:60]}...)"

        return response

    def _mock_opening_claim(self, agent_config: AgentModel, topic: str) -> str:
        return self._generate_topic_aware_mock(topic, agent_config.role, "claim")

    def _mock_response(
        self,
        agent_config: AgentModel,
        topic: str,
        messages: List[Message],
        feedback: str | None,
    ) -> str:
        last_type = messages[-1].message_type if messages else "claim"
        if agent_config.role == "opponent" or last_type in ("claim", "rebuttal"):
            msg_type = "attack"
        else:
            msg_type = "rebuttal"
        return self._generate_topic_aware_mock(topic, agent_config.role, msg_type, feedback)
