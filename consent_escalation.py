import logging
import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    AgentTask,
    JobContext,
    RunContext,
    RoomInputOptions,
    WorkerOptions,
    get_job_context,
    cli,
)
from livekit.agents.llm import function_tool, ToolError
from livekit.plugins import openai, deepgram, cartesia, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger(__name__)
load_dotenv()


# ─────────────────────────────────────────────
# 1. Consent task — runs before main assistant
# ─────────────────────────────────────────────
class CollectConsent(AgentTask):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions=(
                "Ask for recording consent and get a clear yes or no answer. "
                "Be polite and professional. Once you have a clear answer, "
                "call consent_given or consent_denied immediately."
            ),
            chat_ctx=chat_ctx,
        )
        self._result: bool = False

    @property
    def result(self) -> bool:
        return self._result

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Briefly introduce yourself as a voice assistant, then ask for permission "
                "to record the call for quality assurance and training purposes. "
                "Make it clear they can decline."
            )
        )

    @function_tool
    async def consent_given(self) -> None:
        """Call this when the user clearly gives consent to record the call."""
        logger.info("✅ Consent given")
        self._result = True
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Call this when the user clearly declines consent to record the call."""
        logger.info("🚫 Consent denied")
        self._result = False
        self.complete(False)


# ─────────────────────────────────────────────
# 2. Manager agent — handles escalations
# ─────────────────────────────────────────────
class Manager(Agent):
    def __init__(self, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                "You are a senior manager for a voice AI tech support team. "
                "A customer has been escalated to you. "
                "Be empathetic, professional, and solution-focused. "
                "Listen to their issue and offer a concrete resolution or next steps. "
                "Keep replies under 3 sentences."
            ),
            tts=cartesia.TTS(
                model="sonic-3",
                voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as the manager who has taken over the call. "
                "Acknowledge the customer's concern briefly and ask how you can help resolve it."
            )
        )


# ─────────────────────────────────────────────
# 3. Main assistant
# ─────────────────────────────────────────────
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, keep replies under 3 sentences. "
                "You can look up weather if asked. "
                "You can answer questions about LiveKit by searching the documentation. "
                "If the user asks to speak to a manager or escalate, use escalate_to_manager."
            ),
        )

    async def on_enter(self) -> None:
        # Run consent task — waits for completion before proceeding
        task = CollectConsent(chat_ctx=self.chat_ctx)
        consent = await task

        if consent:
            logger.info("🟢 Consent granted — starting main session")
            await self.session.generate_reply(
                instructions="Thank the user warmly and offer your assistance."
            )
        else:
            logger.info("🔴 Consent denied — ending call")
            await self.session.generate_reply(
                instructions=(
                    "Politely inform the user that without recording consent you are unable "
                    "to proceed, thank them for calling, and say goodbye."
                )
            )
            await get_job_context().shutdown()

    # ── Tools ────────────────────────────────

    @function_tool
    async def escalate_to_manager(self, context: RunContext) -> tuple:
        """Escalate the call to a manager when the user requests it."""
        logger.info("📞 Escalating call to manager")
        return Manager(chat_ctx=self.chat_ctx), "Escalating you to my manager now."

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> dict:
        """Look up the current weather for a given location.

        Args:
            location: City name or location to get weather for.
        """
        logger.info("🌤️  Looking up weather for %s", location)
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                raise ToolError(f"Could not find location: {location}")

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            place_name = geo_data["results"][0]["name"]

            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",
                    "temperature_unit": "fahrenheit",
                },
            )
            weather = weather_response.json()
            temp = weather["current"]["temperature_2m"]
            code = weather["current"]["weather_code"]

            logger.info("✅ Weather fetched for %s: %s°F (code %s)", place_name, temp, code)
            return {"location": place_name, "temperature": temp, "conditions": code}

    @function_tool
    async def search_livekit_docs(self, context: RunContext, query: str) -> str:
        """Search the LiveKit documentation to answer questions about LiveKit features,
        APIs, SDKs, or how to build with LiveKit.

        Args:
            query: The search query or question about LiveKit.
        """
        import urllib.parse
        from html.parser import HTMLParser

        class DDGParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self._current = {}
                self._capture = None

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and "result__a" in attrs.get("class", ""):
                    self._current["url"] = attrs.get("href", "")
                    self._capture = "title"
                elif tag == "a" and "result__snippet" in attrs.get("class", ""):
                    self._capture = "snippet"

            def handle_data(self, data):
                if self._capture:
                    self._current[self._capture] = self._current.get(self._capture, "") + data

            def handle_endtag(self, tag):
                if tag == "a" and self._capture == "snippet":
                    if self._current.get("url") and self._current.get("snippet"):
                        self.results.append(dict(self._current))
                    self._current = {}
                    self._capture = None

        logger.info("📚 Searching LiveKit docs for: %s", query)
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(f"site:docs.livekit.io {query}")

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            try:
                response = await client.get(
                    url, headers={"User-Agent": "Mozilla/5.0 (compatible; voice-agent/1.0)"}
                )
                if response.status_code == 200:
                    parser = DDGParser()
                    parser.feed(response.text)
                    results = parser.results[:3]
                    if not results:
                        return f"No LiveKit docs results found for '{query}'. Check https://docs.livekit.io"
                    summaries = [
                        f"{r.get('title','').strip()}: {r.get('snippet','').strip()[:300]} (source: {r.get('url','')})"
                        for r in results
                    ]
                    logger.info("✅ LiveKit docs search returned %d results", len(results))
                    return "\n\n".join(summaries)

                return "I wasn't able to search the docs right now. Check https://docs.livekit.io"

            except Exception as exc:
                logger.error("❌ LiveKit docs search failed: %s", exc)
                raise ToolError(f"Failed to search LiveKit documentation: {exc}")


# ─────────────────────────────────────────────
# 4. Entrypoint
# ─────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    vad = silero.VAD.load()

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt=deepgram.STT(),
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=vad,
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        num_idle_processes=1,
    ))