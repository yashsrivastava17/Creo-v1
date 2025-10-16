from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Awaitable, Callable, Protocol
from uuid import uuid4

from jarvis.llm.types import RoutingPlan
from jarvis.orchestrator.events import AssistantMessage, State, UserTurn, WakeWordHit
from jarvis.persona import PersonaManager
from jarvis.telemetry.logging import get_logger


class MemoryService(Protocol):
    async def recall(self, user_turn: UserTurn | None) -> dict: ...

    async def guard(self, memory_writes: list[dict]) -> list[dict]: ...

    async def write(self, writes: list[dict]) -> None: ...


class RouterService(Protocol):
    async def decide(self, prompt_text: str, context: dict) -> RoutingPlan: ...


class TTSService(Protocol):
    async def speak(self, turn_id: str, text: str) -> Awaitable[float]: ...


class ToolExecutor(Protocol):
    async def invoke(self, call: dict) -> dict: ...


class FloatingUIBridge(Protocol):
    async def publish_state(self, state: State, payload: dict | None = None) -> None: ...


class Orchestrator:
    def __init__(
        self,
        memory: MemoryService,
        router: RouterService,
        tts: TTSService,
        persona: PersonaManager,
        tool_executor: ToolExecutor,
        ui_bridge: FloatingUIBridge,
    ) -> None:
        self._memory = memory
        self._router = router
        self._tts = tts
        self._persona = persona
        self._tool_executor = tool_executor
        self._ui = ui_bridge
        self._logger = get_logger(__name__)
        self._state: State = "IDLE"
        self._turn_lock = asyncio.Lock()
        self._agent_counter = 0

    @property
    def state(self) -> State:
        return self._state

    async def set_state(self, state: State, payload: dict | None = None) -> None:
        self._state = state
        self._logger.debug("state.transition", state=state, payload=payload)
        await self._ui.publish_state(state, payload=payload or {})

    async def handle_turn(
        self,
        transcript: str,
        wake_hit: WakeWordHit,
        audio_refs: list[str],
        context_factory: Callable[[], Awaitable[dict]],
    ) -> float:
        async with self._turn_lock:
            turn_id = str(uuid4())
            persona_profile = self._persona.active_profile()
            await self.set_state("COMPOSING", {"turn_id": turn_id, "persona": persona_profile})
            await self._publish_agent_step("context", "running")
            context = await context_factory()
            await self._publish_agent_step("context", "complete")
            user_turn = UserTurn(
                turn_id=turn_id,
                wake_word_ts=wake_hit.ts,
                prompt_text=transcript,
                context=context,
                audio_refs=audio_refs,
                tools_allowed=context.get("tools_allowed", []),
            )

            await self._publish_agent_step("router_decide", "running")
            plan = await self._router.decide(user_turn.prompt_text, user_turn.context)
            await self._publish_agent_step("router_decide", "complete", provider=plan.provider.name)
            await self._ui.publish_state("COMPOSING", {"turn_id": turn_id, "plan": plan.to_dict()})

            async with AsyncExitStack() as stack:
                await self._publish_agent_step("llm_generate", "running", provider=plan.provider.name)
                resp = await plan.provider.chat(plan.prompt, tools=plan.tools)
                await self._publish_agent_step("llm_generate", "complete", provider=plan.provider.name)

                tool_results: list[dict] = []
                for call in resp.tool_calls:
                    result = await self._tool_executor.invoke(call)
                    tool_results.append(result)
                    await self._publish_agent_step("tool_call", "complete", detail={"tool": call.get("tool")})

                if tool_results:
                    followup_prompt = plan.followup_prompt(tool_results)
                    await self._publish_agent_step("llm_followup", "running", provider=plan.provider.name)
                    resp = await plan.provider.chat(followup_prompt, tools=plan.tools)
                    await self._publish_agent_step("llm_followup", "complete", provider=plan.provider.name)

                gated = await self._memory.guard(resp.memory_writes)
            if gated:
                await self._memory.write(gated)

            await self.set_state(
                "SPEAKING",
                {
                    "turn_id": turn_id,
                    "text": resp.text,
                    "persona": persona_profile,
                    "tts_status": "starting",
                },
            )
            duration = await self._tts.speak(turn_id, resp.text)
            await self._ui.publish_state(
                "SPEAKING",
                {
                    "turn_id": turn_id,
                    "text": resp.text,
                    "persona": persona_profile,
                    "tts_status": "complete",
                    "tts_duration": duration,
                },
            )

        await self.set_state("IDLE", {"turn_id": turn_id})
        return duration

    async def run_text(self, transcript: str, context_factory: Callable[[], Awaitable[dict]]) -> AssistantMessage:
        async with self._turn_lock:
            turn_id = str(uuid4())
            persona_profile = self._persona.active_profile()
            await self._ui.publish_state(
                "COMPOSING",
                {"turn_id": turn_id, "transcript": transcript, "mode": "text", "persona": persona_profile},
            )
            await self._publish_agent_step("context", "running")
            context = await context_factory()
            await self._publish_agent_step("context", "complete")

            await self._publish_agent_step("router_decide", "running")
            plan = await self._router.decide(transcript, context)
            await self._publish_agent_step("router_decide", "complete", provider=plan.provider.name)
            await self._ui.publish_state(
                "COMPOSING",
                {"turn_id": turn_id, "plan": plan.to_dict(), "transcript": transcript, "persona": persona_profile},
            )

            await self._publish_agent_step("llm_generate", "running", provider=plan.provider.name)
            resp = await plan.provider.chat(plan.prompt, tools=plan.tools)
            await self._publish_agent_step("llm_generate", "complete", provider=plan.provider.name)

            tool_results: list[dict] = []
            for call in resp.tool_calls:
                result = await self._tool_executor.invoke(call)
                tool_results.append(result)
                await self._publish_agent_step("tool_call", "complete", detail={"tool": call.get("tool")})

            if tool_results:
                followup_prompt = plan.followup_prompt(tool_results)
                await self._publish_agent_step("llm_followup", "running", provider=plan.provider.name)
                resp = await plan.provider.chat(followup_prompt, tools=plan.tools)
                await self._publish_agent_step("llm_followup", "complete", provider=plan.provider.name)

            gated = await self._memory.guard(resp.memory_writes)
            if gated:
                await self._memory.write(gated)

            message = AssistantMessage(
                turn_id=turn_id,
                text=resp.text,
                citations=[],
                memory_writes=resp.memory_writes,
                tts_duration=duration,
            )
            await self._ui.publish_state(
                "SPEAKING",
                {
                    "turn_id": turn_id,
                    "text": resp.text,
                    "mode": "text",
                    "persona": persona_profile,
                    "tts_status": "starting",
                },
            )
            duration = await self._tts.speak(turn_id, resp.text)
            await self._ui.publish_state(
                "SPEAKING",
                {
                    "turn_id": turn_id,
                    "text": resp.text,
                    "mode": "text",
                    "persona": persona_profile,
                    "tts_status": "complete",
                    "tts_duration": duration,
                },
            )
            await self._ui.publish_state("IDLE", {"turn_id": turn_id})
            return message

    async def _publish_agent_step(
        self,
        name: str,
        status: str,
        provider: str | None = None,
        detail: dict | None = None,
    ) -> None:
        self._agent_counter += 1
        payload = {
            "id": self._agent_counter,
            "name": name,
            "status": status,
            "provider": provider,
        }
        if detail:
            payload.update(detail)
        await self._ui.publish_state("LLM_AGENT", payload)


__all__ = ["Orchestrator"]
