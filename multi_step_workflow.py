"""
Multi-step voice agent workflow:
  Step 1 — Onboarding   : Welcome user, collect recording consent
  Step 2 — Data Collection: Gather name, issue type, and description
  Step 3 — Handoff       : Route to the right specialist agent (Billing / Tech / General)
"""

import logging
import httpx
from dataclasses import dataclass, field
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


# ─────────────────────────────────────────────────────────────
# Shared session data — passed between tasks
# ─────────────────────────────────────────────────────────────
@dataclass
class CustomerProfile:
    consent: bool = False
    name: str = ""
    issue_type: str = ""        # "billing" | "tech" | "general"
    issue_description: str = ""


# ─────────────────────────────────────────────────────────────
# STEP 1 — Onboarding Task
# Welcome the user and collect recording consent
# ─────────────────────────────────────────────────────────────
class OnboardingTask(AgentTask):
    def __init__(self, profile: CustomerProfile, chat_ctx=None):
        super().__init__(
            instructions=(
                "You are a friendly onboarding assistant. "
                "Welcome the caller warmly and ask for their permission to record "
                "the call for quality and training purposes. "
                "Once you have a clear yes or no, call the appropriate tool immediately."
            ),
            chat_ctx=chat_ctx,
        )
        self.profile = profile

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Welcome the caller to support. Tell them this call may be recorded "
                "for quality assurance and ask if they consent. Keep it warm and brief."
            )
        )

    @function_tool
    async def consent_given(self) -> None:
        """Call this when the user agrees to be recorded."""
        logger.info("✅ [Onboarding] Consent granted")
        self.profile.consent = True
        self.complete(True)

    @function_tool
    async def consent_denied(self) -> None:
        """Call this when the user refuses to be recorded."""
        logger.info("🚫 [Onboarding] Consent denied — ending call")
        self.profile.consent = False
        self.complete(False)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Data Collection Task
# Gather name, issue type, and a brief description
# ─────────────────────────────────────────────────────────────
class DataCollectionTask(AgentTask):
    def __init__(self, profile: CustomerProfile, chat_ctx=None):
        super().__init__(
            instructions=(
                "You are a helpful intake assistant collecting information before "
                "routing the caller to the right team. "
                "Collect the caller's name, the type of issue (billing, technical, or general), "
                "and a brief description of the problem. "
                "Once you have all three pieces of information, call submit_info."
            ),
            chat_ctx=chat_ctx,
        )
        self.profile = profile

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Thank the user for their consent. Tell them you need to gather "
                "a little information before connecting them with the right team. "
                "Start by asking for their name."
            )
        )

    @function_tool
    async def submit_info(
        self,
        name: str,
        issue_type: str,
        issue_description: str,
    ) -> None:
        """Submit the collected customer information.

        Args:
            name: The caller's full name.
            issue_type: One of 'billing', 'tech', or 'general'.
            issue_description: A brief description of the caller's issue.
        """
        self.profile.name = name
        self.profile.issue_type = issue_type.lower().strip()
        self.profile.issue_description = issue_description

        logger.info(
            "📋 [DataCollection] Profile complete — name=%s, type=%s, issue=%s",
            name, issue_type, issue_description,
        )
        self.complete(True)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Specialist Agents (Handoff targets)
# ─────────────────────────────────────────────────────────────
class BillingSpecialist(Agent):
    def __init__(self, profile: CustomerProfile, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                f"You are a billing specialist. "
                f"The caller's name is {profile.name}. "
                f"Their issue: {profile.issue_description}. "
                "Help them resolve billing questions: charges, invoices, refunds, or subscriptions. "
                "Be clear, empathetic, and solution-focused. Keep replies under 3 sentences."
            ),
            tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"Introduce yourself as the billing specialist. "
                f"Address {self.profile_name()} by name, acknowledge their billing issue, "
                "and ask how you can help resolve it."
            )
        )

    def profile_name(self):
        # helper to avoid closure issues
        for kw in self.__dict__.values():
            if isinstance(kw, CustomerProfile):
                return kw.name
        return "there"


class TechSpecialist(Agent):
    def __init__(self, profile: CustomerProfile, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                f"You are a technical support specialist. "
                f"The caller's name is {profile.name}. "
                f"Their issue: {profile.issue_description}. "
                "Help them troubleshoot technical problems step by step. "
                "Be concise and practical. Keep replies under 3 sentences."
            ),
            tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"Introduce yourself as the tech support specialist. "
                f"Greet the caller by name, acknowledge their technical issue, "
                "and ask one clarifying question to start diagnosing."
            )
        )


class GeneralSupport(Agent):
    def __init__(self, profile: CustomerProfile, chat_ctx=None) -> None:
        super().__init__(
            instructions=(
                f"You are a general support agent. "
                f"The caller's name is {profile.name}. "
                f"Their issue: {profile.issue_description}. "
                "Help them with general enquiries, account questions, or anything else. "
                "Be helpful and friendly. Keep replies under 3 sentences."
            ),
            tts=cartesia.TTS(model="sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as the support agent. "
                "Greet the caller by name and let them know you're here to help "
                "with whatever they need."
            )
        )


# ─────────────────────────────────────────────────────────────
# Orchestrator Agent — runs all steps and routes to specialist
# ─────────────────────────────────────────────────────────────
class Orchestrator(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are an orchestrator managing a multi-step support workflow.",
        )
        self.profile = CustomerProfile()
        self._specialist = None

    async def on_enter(self) -> None:
        # ── Step 1: Onboarding & consent ──────────────────────
        logger.info("🚀 Step 1: Onboarding")
        consent = await OnboardingTask(self.profile, chat_ctx=self.chat_ctx)

        if not consent:
            await self.session.generate_reply(
                instructions=(
                    "Politely inform the caller that without recording consent "
                    "you cannot proceed. Thank them and say goodbye."
                )
            )
            await get_job_context().shutdown()
            return

        # ── Step 2: Data collection ────────────────────────────
        logger.info("🚀 Step 2: Data collection")
        await DataCollectionTask(self.profile, chat_ctx=self.chat_ctx)

        logger.info(
            "📦 Profile: name=%s | type=%s | issue=%s",
            self.profile.name,
            self.profile.issue_type,
            self.profile.issue_description,
        )

        # ── Step 3: Route to specialist via function_tool ────
        logger.info("🚀 Step 3: Handoff → %s", self.profile.issue_type)

        if self.profile.issue_type == "billing":
            self._specialist = BillingSpecialist(self.profile, chat_ctx=self.chat_ctx)
        elif self.profile.issue_type in ("tech", "technical"):
            self._specialist = TechSpecialist(self.profile, chat_ctx=self.chat_ctx)
        else:
            self._specialist = GeneralSupport(self.profile, chat_ctx=self.chat_ctx)

        # Trigger LLM to call the transfer tool (same pattern as manager escalation)
        await self.session.generate_reply(
            instructions=(
                f"Tell {self.profile.name} you have everything you need and are now "
                "connecting them with the right specialist. Then call transfer_to_specialist."
            )
        )

    @function_tool
    async def transfer_to_specialist(self, context: RunContext):
        """Transfer the caller to the appropriate specialist agent."""
        logger.info("🔀 Transferring to specialist: %s", type(self._specialist).__name__)
        return self._specialist, f"Connecting you now, {self.profile.name}."


# ─────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────
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
        agent=Orchestrator(),
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