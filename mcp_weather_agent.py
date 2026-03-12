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
                "You can also look up weather if the user asks."
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
            # Step 1: geocode the location
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

            # Step 2: fetch weather (note: param is temperature_2m not temperature_2)
            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,weather_code",  # fixed typo
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