from __future__ import annotations

import asyncio
import re
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Protocol
from uuid import uuid4

from jarvis.llm.types import RoutingPlan
from jarvis.tools import reminders as reminder_tools
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
        self._last_plan: dict[str, Any] | None = None

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
            intent_result = await self._maybe_handle_intent(transcript, turn_id, persona_profile)
            if intent_result is not None:
                duration, _ = intent_result
                await self.set_state("IDLE", {"turn_id": turn_id})
                return duration
            user_turn = UserTurn(
                turn_id=turn_id,
                wake_word_ts=wake_hit.ts,
                prompt_text=transcript,
                context=context,
                audio_refs=audio_refs,
                tools_allowed=context.get("tools_allowed", []),
            )

            await self._publish_agent_step("router_decide", "running")
            routing_plan = await self._router.decide(user_turn.prompt_text, user_turn.context)
            await self._publish_agent_step("router_decide", "complete", provider=routing_plan.provider.name)
            await self._ui.publish_state(
                "COMPOSING",
                {"turn_id": turn_id, "plan": routing_plan.to_dict()},
            )

            plan_results = await self._execute_plan_steps(routing_plan, user_turn)
            if plan_results:
                await self._ui.publish_state(
                    "COMPOSING",
                    {"turn_id": turn_id, "plan_results": plan_results},
                )

            async with AsyncExitStack() as stack:
                await self._publish_agent_step("llm_generate", "running", provider=routing_plan.provider.name)
                composed_prompt = routing_plan.compose_prompt(plan_results)
                resp = await routing_plan.provider.chat(composed_prompt, tools=routing_plan.tools)
                self._router.register_cost(routing_plan.provider.name)
                await self._publish_agent_step("llm_generate", "complete", provider=routing_plan.provider.name)

                tool_results: list[dict] = []
                for call in resp.tool_calls:
                    result = await self._tool_executor.invoke(call)
                    tool_results.append(result)
                    await self._publish_agent_step("tool_call", "complete", detail={"tool": call.get("tool")})

                if tool_results:
                    followup_prompt = routing_plan.followup_prompt(tool_results)
                    await self._publish_agent_step("llm_followup", "running", provider=routing_plan.provider.name)
                    resp = await routing_plan.provider.chat(followup_prompt, tools=routing_plan.tools)
                    self._router.register_cost(routing_plan.provider.name)
                    await self._publish_agent_step("llm_followup", "complete", provider=routing_plan.provider.name)

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
            intent_result = await self._maybe_handle_intent(transcript, turn_id, persona_profile)
            if intent_result is not None:
                duration, text = intent_result
                await self._ui.publish_state("IDLE", {"turn_id": turn_id})
                return AssistantMessage(
                    turn_id=turn_id,
                    text=text,
                    citations=[],
                    memory_writes=[],
                    tts_duration=duration,
                )

            await self._publish_agent_step("router_decide", "running")
            routing_plan = await self._router.decide(transcript, context)
            await self._publish_agent_step("router_decide", "complete", provider=routing_plan.provider.name)
            await self._ui.publish_state(
                "COMPOSING",
                {
                    "turn_id": turn_id,
                    "plan": routing_plan.to_dict(),
                    "transcript": transcript,
                    "persona": persona_profile,
                },
            )

            plan_results = await self._execute_plan_steps(routing_plan, None)
            if plan_results:
                await self._ui.publish_state(
                    "COMPOSING",
                    {"turn_id": turn_id, "plan_results": plan_results, "transcript": transcript},
                )

            await self._publish_agent_step("llm_generate", "running", provider=routing_plan.provider.name)
            composed_prompt = routing_plan.compose_prompt(plan_results)
            resp = await routing_plan.provider.chat(composed_prompt, tools=routing_plan.tools)
            self._router.register_cost(routing_plan.provider.name)
            await self._publish_agent_step("llm_generate", "complete", provider=routing_plan.provider.name)

            tool_results: list[dict] = []
            for call in resp.tool_calls:
                result = await self._tool_executor.invoke(call)
                tool_results.append(result)
                await self._publish_agent_step("tool_call", "complete", detail={"tool": call.get("tool")})

            if tool_results:
                followup_prompt = routing_plan.followup_prompt(tool_results)
                await self._publish_agent_step("llm_followup", "running", provider=routing_plan.provider.name)
                resp = await routing_plan.provider.chat(followup_prompt, tools=routing_plan.tools)
                self._router.register_cost(routing_plan.provider.name)
                await self._publish_agent_step("llm_followup", "complete", provider=routing_plan.provider.name)

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

    async def _invoke_tool(self, name: str, args: dict[str, Any], origin: str = "intent") -> dict[str, Any]:
        call = {"tool": name, "args": args, "origin": origin}
        return await self._tool_executor.invoke(call)

    def cache_plan(self, plan: dict[str, Any]) -> None:
        self._last_plan = plan

    async def _maybe_handle_intent(
        self,
        transcript: str,
        turn_id: str,
        persona_profile: dict[str, Any],
    ) -> tuple[float, str] | None:
        intent = self._match_intent(transcript)
        if intent is None:
            return None
        await self._publish_agent_step("intent", "running", detail={"intent": intent["name"]})
        response_text = ""
        try:
            if intent["name"] == "plan_day":
                await self._invoke_tool("tasks.sync_from_apple", {"list_names": None})
                today = datetime.now(timezone.utc).date().isoformat()
                plan = await self._invoke_tool(
                    "planner.plan_day",
                    {"date": today, "mode": intent.get("mode")},
                )
                self._last_plan = plan
                await self._ui.publish_state("PLANNER", plan)
                response_text = f"I've drafted your {intent.get('mode', 'balanced')} plan for today."
            elif intent["name"] == "whats_next":
                plan = self._last_plan
                if plan is None:
                    today = datetime.now(timezone.utc).date().isoformat()
                    plan = await self._invoke_tool("planner.plan_day", {"date": today})
                    self._last_plan = plan
                next_block = self._next_block(plan)
                if next_block:
                    response_text = f"Next up at {next_block['start_time']} is {next_block['label']}."
                else:
                    response_text = "No upcoming blocks on your schedule."
            elif intent["name"] == "remind_at":
                when = intent["when"]
                title = intent.get("title", "Reminder")
                await self._invoke_tool(
                    "reminders.add",
                    {"when": when, "title": title, "payload": {}, "channel": "local"},
                )
                response_text = f"Reminder set for {when.strftime('%I:%M %p').lstrip('0')} to {title}."
            elif intent["name"] == "remind_in":
                when = datetime.now(timezone.utc) + intent["delta"]
                title = intent.get("title", "Reminder")
                await self._invoke_tool(
                    "reminders.add",
                    {"when": when, "title": title, "payload": {}, "channel": "local"},
                )
                minutes = int(intent["delta"].total_seconds() // 60)
                response_text = f"Okay, I'll remind you in {minutes} minutes."
            elif intent["name"] == "router_mode":
                state = await self._invoke_tool("router.set_mode", {"smart": intent["smart"]})
                provider = state.get("provider", "your default model")
                response_text = f"Routing future calls through {provider}."
            elif intent["name"] == "prep_event":
                plan = self._last_plan
                if plan is None:
                    today = datetime.now(timezone.utc).date().isoformat()
                    plan = await self._invoke_tool("planner.plan_day", {"date": today})
                    self._last_plan = plan
                target = self._find_event_by_time(plan, intent["hour"], intent["minute"])
                if target:
                    prep_time = target["start"] - timedelta(minutes=15)
                    await self._invoke_tool(
                        "reminders.add",
                        {
                            "when": prep_time,
                            "title": f"Prep for {target['label']}",
                            "payload": {"event_id": target["event_id"]},
                        },
                    )
                    response_text = f"I'll remind you fifteen minutes before {target['label']}."
                else:
                    response_text = "I couldn't find that meeting on your plan yet."
            elif intent["name"] == "start_notes":
                event = self._current_event_from_plan()
                if event is None:
                    response_text = "I don't see an active meeting to start notes for."
                else:
                    await self._invoke_tool("meeting.start_notes", {"event_id": event["event_id"]})
                    response_text = f"Starting notes for {event['label']}."
            elif intent["name"] == "snooze_reminder":
                reminder = self._next_reminder()
                if reminder is None:
                    response_text = "There's no reminder to snooze."
                else:
                    reminder_tools.SCHEDULER.cancel(reminder["id"])
                    new_time = datetime.fromisoformat(reminder["when"]) + intent["delta"]
                    reminder_tools.SCHEDULER.add(
                        when_iso=new_time.isoformat(),
                        title=reminder.get("title", "Reminder"),
                        payload=reminder.get("payload", {}),
                        channel=reminder.get("channel", "local"),
                    )
                    response_text = f"Snoozed for {int(intent['delta'].total_seconds() // 60)} minutes."
            else:
                response_text = ""
            await self._publish_agent_step("intent", "complete", detail={"intent": intent["name"]})
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._logger.error("intent.handler.failed", intent=intent["name"], error=str(exc))
            await self._publish_agent_step("intent", "error", detail={"error": str(exc)})
            response_text = "Sorry, that action failed."

        if not response_text:
            return None

        await self.set_state(
            "SPEAKING",
            {
                "turn_id": turn_id,
                "text": response_text,
                "persona": persona_profile,
                "tts_status": "starting",
            },
        )
        duration = await self._tts.speak(turn_id, response_text)
        await self._ui.publish_state(
            "SPEAKING",
            {
                "turn_id": turn_id,
                "text": response_text,
                "persona": persona_profile,
                "tts_status": "complete",
                "tts_duration": duration,
            },
        )
        return duration, response_text

    def _match_intent(self, transcript: str) -> dict[str, Any] | None:
        text = transcript.strip().lower().replace("’", "'")
        if not text:
            return None
        if "plan my day" in text:
            mode = "balanced"
            if "deep" in text:
                mode = "deep"
            elif "speed" in text:
                mode = "speed"
            return {"name": "plan_day", "mode": mode}
        if "what's next" in text or "whats next" in text:
            return {"name": "whats_next"}
        if "be smarter" in text:
            return {"name": "router_mode", "smart": True}
        if "be lighter" in text:
            return {"name": "router_mode", "smart": False}
        match = re.search(r"prep me for (?:the )?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<suffix>am|pm)?", text)
        if match:
            hour = int(match.group("hour"))
            minute = int(match.group("minute") or 0)
            suffix = match.group("suffix")
            hour = self._to_24_hour(hour, suffix)
            return {"name": "prep_event", "hour": hour, "minute": minute}
        if "start notes for this meeting" in text or text.startswith("i'm entering a meeting") or text.startswith("im entering a meeting"):
            return {"name": "start_notes"}
        match = re.search(r"snooze (?:the )?next reminder (?P<amount>\d+)\s*(?P<unit>minutes?|minute|hours?|hour)", text)
        if match:
            amount = int(match.group("amount"))
            unit = match.group("unit")
            delta = timedelta(minutes=amount) if "hour" not in unit else timedelta(hours=amount)
            return {"name": "snooze_reminder", "delta": delta}
        match = re.search(
            r"remind me at (?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<suffix>am|pm)?\s*(?:to)?\s*(?P<what>.+)",
            text,
        )
        if match:
            hour = self._to_24_hour(int(match.group("hour")), match.group("suffix"))
            minute = int(match.group("minute") or 0)
            title = match.group("what").strip()
            when = datetime.now(timezone.utc).replace(hour=hour, minute=minute, second=0, microsecond=0)
            if when <= datetime.now(timezone.utc):
                when += timedelta(days=1)
            return {"name": "remind_at", "when": when, "title": title}
        match = re.search(
            r"remind me in (?P<amount>\d+)\s*(?P<unit>minutes?|minute|hours?|hour)\s*(?:to)?\s*(?P<what>.+)",
            text,
        )
        if match:
            amount = int(match.group("amount"))
            unit = match.group("unit")
            delta = timedelta(minutes=amount) if "hour" not in unit else timedelta(hours=amount)
            title = match.group("what").strip()
            return {"name": "remind_in", "delta": delta, "title": title}
        return None

    @staticmethod
    def _to_24_hour(hour: int, suffix: str | None) -> int:
        if suffix is None:
            return hour % 24
        suffix = suffix.lower()
        if suffix == "pm" and hour != 12:
            return hour + 12
        if suffix == "am" and hour == 12:
            return 0
        return hour

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _next_block(self, plan: dict[str, Any]) -> dict[str, str] | None:
        now = datetime.now(timezone.utc)
        for block in plan.get("blocks", []):
            start = self._parse_dt(block.get("start"))
            if start and start >= now:
                return {
                    "label": block.get("label", ""),
                    "start_time": start.strftime("%I:%M %p").lstrip("0"),
                }
        return None

    def _find_event_by_time(self, plan: dict[str, Any], hour: int, minute: int) -> dict[str, Any] | None:
        for block in plan.get("blocks", []):
            if block.get("type") != "event":
                continue
            start = self._parse_dt(block.get("start"))
            if start and start.hour == hour and start.minute == minute:
                return {
                    "label": block.get("label", "meeting"),
                    "start": start,
                    "event_id": block.get("meta", {}).get("id"),
                }
        return None

    def _current_event_from_plan(self) -> dict[str, Any] | None:
        if self._last_plan is None:
            return None
        now = datetime.now(timezone.utc)
        for block in self._last_plan.get("blocks", []):
            if block.get("type") != "event":
                continue
            start = self._parse_dt(block.get("start"))
            end = self._parse_dt(block.get("end"))
            if start and end and start <= now <= end:
                return {
                    "label": block.get("label", "meeting"),
                    "event_id": block.get("meta", {}).get("id"),
                }
        return None

    def _next_reminder(self) -> dict[str, Any] | None:
        active = reminder_tools.SCHEDULER.list_active()
        if not active:
            return None
        return min(active, key=lambda reminder: reminder.get("when", ""))

    async def _execute_plan_steps(self, plan: RoutingPlan, user_turn: UserTurn | None) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for step in plan.plan.steps:
            if step.use == "respond":
                continue
            detail = {"tool": step.use}
            await self._publish_agent_step("plan_step", "running", detail=detail)
            call_payload = {
                "tool": step.use,
                "args": step.args,
                "origin": "plan",
                "turn_id": user_turn.turn_id if user_turn else None,
            }
            try:
                result = await self._tool_executor.invoke(call_payload)
                results.append({"tool": step.use, "args": step.args, "result": result})
                await self._publish_agent_step("plan_step", "complete", detail=detail)
            except Exception as exc:  # pragma: no cover - surface to logging/telemetry
                self._logger.error("plan.step.failed", tool=step.use, error=str(exc))
                results.append({"tool": step.use, "args": step.args, "error": str(exc)})
                await self._publish_agent_step("plan_step", "error", detail=detail | {"error": str(exc)})
        return results


__all__ = ["Orchestrator"]
