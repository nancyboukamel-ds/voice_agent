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
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics

logger = logging.getLogger(__name__)
load_dotenv()

# ─────────────────────────────────────────────
# STT accuracy proxy: phrases that suggest the
# user is correcting a mis-transcription
# ─────────────────────────────────────────────
CORRECTION_PHRASES = [
    "no i said", "i said", "that's wrong", "not that",
    "i meant", "no i meant", "wrong", "that's not what i said",
    "no that's not", "you misheard", "try again", "repeat that",
    "i didn't say", "not what i said",
]


# ─────────────────────────────────────────────
# Tracked FallbackAdapter wrappers
# These wrap each FallbackAdapter so we can log
# when it switches to a backup provider
# ─────────────────────────────────────────────
class TrackedLLMFallback(llm.FallbackAdapter):
    def __init__(self, instances, fallback_counter: dict):
        super().__init__(instances)
        self._counter = fallback_counter
        self._active_index = 0
        # hook into the fallback event
        self.on("fallback", self._on_fallback)

    def _on_fallback(self, ev):
        self._active_index += 1
        self._counter["llm"] += 1
        logger.warning(
            "⚠️  LLM fallback activated! Switching to backup provider. "
            "Total LLM fallbacks: %d", self._counter["llm"]
        )


class TrackedSTTFallback(stt.FallbackAdapter):
    def __init__(self, instances, fallback_counter: dict):
        super().__init__(instances)
        self._counter = fallback_counter
        self.on("fallback", self._on_fallback)

    def _on_fallback(self, ev):
        self._counter["stt"] += 1
        logger.warning(
            "⚠️  STT fallback activated! Switching to backup provider. "
            "Total STT fallbacks: %d", self._counter["stt"]
        )


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences."
            ),
        )


async def entrypoint(ctx: JobContext):
    vad = silero.VAD.load()
    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    # ── Counters ──────────────────────────────
    interruption_count = 0
    fallback_counter = {"llm": 0, "stt": 0}

    # STT accuracy proxy counters
    stt_accuracy = {
        "correction_requests": 0,   # user said "i meant / that's wrong / etc."
        "intent_reversals": 0,      # user said "no" right after agent response
        "total_turns": 0,
    }

    # ── Session ───────────────────────────────
    session = AgentSession(
        llm=TrackedLLMFallback(
            [
                openai.LLM(model="gpt-4o-mini"),
                openai.LLM(model="gpt-4o"),
            ],
            fallback_counter,
        ),
        stt=TrackedSTTFallback(
            [
                deepgram.STT(),
            ],
            fallback_counter,
        ),
        tts=cartesia.TTS(
            model="sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        vad=vad,
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )

    # ── Metrics event ─────────────────────────
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        print("📊 metrics collected:", ev.metrics)
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

    # ── Agent state event ─────────────────────
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        nonlocal interruption_count

        # barge-in: user started speaking while agent was speaking
        if ev.new_state == "listening" and session.current_speech:
            interruption_count += 1
            logger.info(
                "⚡ Barge-in detected! Total interruptions: %d", interruption_count
            )

        # TTFA: log when agent starts speaking (use tts_ttfb from TTS metrics instead)
        if ev.new_state == "speaking":
            logger.info("🔊 Agent started speaking (see tts_ttfb in TTS metrics for TTFA)")

    # ── STT accuracy proxy: analyse each user transcript ──
    @session.on("user_input_transcribed")
    def _on_transcript(ev):
        nonlocal stt_accuracy
        text = ev.transcript.lower().strip()
        stt_accuracy["total_turns"] += 1

        # correction request: user explicitly corrects transcription
        if any(phrase in text for phrase in CORRECTION_PHRASES):
            stt_accuracy["correction_requests"] += 1
            logger.warning(
                "🔤 STT correction detected: '%s'  "
                "Total corrections: %d",
                ev.transcript,
                stt_accuracy["correction_requests"],
            )

        # intent reversal: user starts turn with "no" (handles "no.", "no,", "no!")
        import re
        if re.match(r'^no[\s.,!?]', text) or text == "no":
            stt_accuracy["intent_reversals"] += 1
            logger.warning(
                "↩️  Intent reversal detected: '%s'  "
                "Total reversals: %d",
                ev.transcript,
                stt_accuracy["intent_reversals"],
            )

    # ── Shutdown summary ──────────────────────
    async def log_usage():
        summary = usage_collector.get_summary()
        total = stt_accuracy["total_turns"] or 1   # avoid div/0
        correction_rate = stt_accuracy["correction_requests"] / total * 100
        reversal_rate = stt_accuracy["intent_reversals"] / total * 100

        logger.info("=" * 60)
        logger.info("📈 SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info("Usage:                  %s", summary)
        logger.info("-" * 60)
        logger.info("⚡ Interruptions:        %d", interruption_count)
        logger.info("-" * 60)
        logger.info("⚠️  LLM fallbacks:        %d", fallback_counter["llm"])
        logger.info("⚠️  STT fallbacks:        %d", fallback_counter["stt"])
        logger.info("-" * 60)
        logger.info("🔤 STT accuracy proxies:")
        logger.info("   Total turns:         %d", stt_accuracy["total_turns"])
        logger.info(
            "   Correction requests: %d (%.1f%%)",
            stt_accuracy["correction_requests"],
            correction_rate,
        )
        logger.info(
            "   Intent reversals:    %d (%.1f%%)",
            stt_accuracy["intent_reversals"],
            reversal_rate,
        )
        logger.info("=" * 60)

    # ── Start session ─────────────────────────
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()
    ctx.add_shutdown_callback(log_usage)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        num_idle_processes=1,
    ))