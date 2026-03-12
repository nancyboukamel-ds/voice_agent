import aiohttp
import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai, deepgram, cartesia, noise_cancellation, silero
from livekit.agents import llm, stt, tts
from livekit.agents.llm import function_tool
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger(__name__)
load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences. "
                "You can also look up weather if the user asks."
            ),
        )

    @function_tool
    async def lookup_weather(self, location: str) -> str:
        """Look up the current weather for a given location."""
        logger.info("🌤️  Looking up weather for %s", location)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://wttr.in/{location}?format=j1",
                    timeout=aiohttp.ClientTimeout(total=10),
                    headers={"User-Agent": "curl/7.68.0"},
                ) as response:
                    if response.status == 200:
                        data = await response.json(content_type=None)
                        if not isinstance(data, dict) or "current_condition" not in data:
                            logger.error("❌ Unexpected response from wttr.in: %s", str(data)[:200])
                            return "Couldn't parse weather data for that location, try a different city name."
                        current = data["current_condition"][0]
                        temp_c = current["temp_C"]
                        temp_f = current["temp_F"]
                        description = current["weatherDesc"][0]["value"]
                        feels_c = current["FeelsLikeC"]
                        humidity = current["humidity"]
                        logger.info(
                            "✅ Weather fetched for %s: %s, %s°C / %s°F",
                            location, description, temp_c, temp_f,
                        )
                        return (
                            f"{description}, {temp_c}°C ({temp_f}°F), "
                            f"feels like {feels_c}°C, humidity {humidity}%"
                        )
                    logger.error("❌ Weather API returned status %s", response.status)
                    return "Weather information is currently unavailable for this location."
        except Exception as exc:
            logger.error("❌ Error fetching weather: %s", exc)
            return "Weather service is temporarily unavailable."


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