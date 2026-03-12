import logging
import httpx
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.llm import function_tool, ToolError
from livekit.plugins import openai, deepgram, cartesia, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger(__name__)
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences. "
                "You can also look up weather if asked. You can also answer questions about "
                "LiveKit by searching the documentation. When users ask about LiveKit "
                "features, APIs, or how to build something, use the docs search tool "
                "to find accurate information."
            ),
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str) -> dict:
        """Look up the current weather for a given location.

        Args:
            location: City name or location to get weather for.
        """
        logger.info("🌤️  Looking up weather for %s", location)
        async with httpx.AsyncClient() as client:
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

            return {
                "location": place_name,
                "temperature": temp,
                "conditions": code,
            }

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
            """Parse DuckDuckGo HTML results to extract titles, snippets, URLs."""
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
        search_query = f"site:docs.livekit.io {query}"
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(search_query)

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            try:
                response = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; voice-agent/1.0)"},
                )
                if response.status_code == 200:
                    parser = DDGParser()
                    parser.feed(response.text)
                    results = parser.results[:3]

                    if not results:
                        logger.warning("⚠️ No docs results found for: %s", query)
                        return f"No LiveKit docs results found for '{query}'. Check https://docs.livekit.io"

                    summaries = []
                    for r in results:
                        title = r.get("title", "").strip()
                        snippet = r.get("snippet", "").strip()[:300]
                        doc_url = r.get("url", "")
                        summaries.append(f"{title}: {snippet} (source: {doc_url})")

                    result_text = "\n\n".join(summaries)
                    logger.info("✅ LiveKit docs search returned %d results", len(results))
                    return result_text

                logger.warning("⚠️ DuckDuckGo returned status %s", response.status_code)
                return "I wasn\'t able to search the docs right now. Check https://docs.livekit.io"

            except Exception as exc:
                logger.error("❌ LiveKit docs search failed: %s", exc)
                raise ToolError(f"Failed to search LiveKit documentation: {exc}")


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