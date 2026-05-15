"""LLM Agent service powered by Pydantic-AI and OpenRouter."""

from __future__ import annotations

import os
import random
from typing import List

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

    def generate_response(
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
        result = agent.run_sync(user_prompt)
        return str(result.output)

    def generate_opening_claim(
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
        result = agent.run_sync(user_prompt)
        return str(result.output)

    # ------------------------------------------------------------------
    # Mock fallback for demo without API keys
    # ------------------------------------------------------------------

    _MOCK_CLAIMS = [
        "Sentetik veri ölçeklendirmesi, sıkı bir ayıklayıcı ağından geçirildiğinde sıfır örnekli görevlerde model performansını güvenilir şekilde artırır. Ampirik sonuçlardaki varyans, öncelikle zayıf filtreleme metodolojilerine bağlanabilir; sentetik veri setlerinin doğasındaki sınırlamalara değil.",
        "Kuantum bilgisayarlama, önümüzdeki beş yıl içinde ilaç keşfinde pratik üstünlüğe ulaşacak ve farmasötik araştırma ile geliştirme takvimini temelden değiştirecektir.",
        "Büyük ölçekli karbon yakalama teknolojisi bugün ekonomik olarak uygulanabilirdir; asıl engel mühendislik kısıtları değil, düzenleyici atalettir.",
    ]

    _MOCK_ATTACKS = [
        "Bu iddia mod çöküşü sorununu aşırı basitleştiriyor. Ayıklayıcılar gürültüyü filtrelerken, sentetik dağılımın çeşitliliğini doğal olarak ayıklayıcının kendi öğrendiği manifold ile sınırlandırırlar. Bu durum, yüksek karmaşıklıktaki akıl yürütme görevlerinde gözlemlenen dağılım uyumsuzluğu ile kanıtlandığı üzere, genellemede azalan getirilere yol açar.",
        "İddia, oda sıcaklığında çözülemeyen koherans kaybı zorluklarını görmezden geliyor. Sentetik moleküller üzerindeki kıyaslama sonuçları henüz gerçek dünya protein katlanma doğruluğuna dönüşmedi.",
        "Ekonomik analiz, yakalama ve depolamanın enerji maliyetini hesaba katmıyor. Son çalışmalar düzleştirilmiş maliyetin karbonun sosyal maliyetinin 3 katı üzerinde olduğunu gösteriyor.",
    ]

    _MOCK_REBUTTALS = [
        "Mod çöküşü eleştirisi eğitim dinamiklerini dağıtım sonuçlarıyla karıştırıyor. arxiv:2305.1234 kaynağındaki ampirik kanıtlar, topluluk ayıklayıcılarının tek model manifold sınırlamasının ötesinde dağılımsal çeşitliliği koruduğunu gösteriyor.",
        "Koherans kaybı gerçekten bir zorluktur, ancak hata düzeltme eşikleri 2022'den bu yana iki büyüklük sırası iyileşti. Yatırımın sürdürülmesi koşuluyla bu eğilim, beş yıllık projeksiyonu destekliyor.",
        "Erken yakalama maliyetleri yüksek olsa da, doğrudan hava yakalama öğrenme eğrileri artık 2010 ile 2020 arasındaki güneş fotovoltaik hücrelerinkini yansıtıyor. Eğilim kesindir.",
    ]

    def _mock_opening_claim(self, agent_config: AgentModel, topic: str) -> str:
        rng = random.Random(hash(topic + str(agent_config.id)))
        return rng.choice(self._MOCK_CLAIMS)

    def _mock_response(
        self,
        agent_config: AgentModel,
        topic: str,
        messages: List[Message],
        feedback: str | None,
    ) -> str:
        rng = random.Random(hash(topic + str(len(messages)) + str(agent_config.id)))
        last_type = messages[-1].message_type if messages else "claim"

        if agent_config.role == "opponent" or last_type in ("claim", "rebuttal"):
            return rng.choice(self._MOCK_ATTACKS)
        return rng.choice(self._MOCK_REBUTTALS)
