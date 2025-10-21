from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jarvis.tools.registry import ToolContext, ToolRegistry, ToolSpec


class TaskNode(BaseModel):
    id: str
    use: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class TaskGraphArgs(BaseModel):
    nodes: list[TaskNode] = Field(default_factory=list)


async def _run_taskgraph(args: TaskGraphArgs, ctx: ToolContext) -> dict[str, Any]:
    # Minimal stub: echo back the plan for now.
    return {
        "executed": [node.model_dump() for node in args.nodes],
        "status": "noop",
        "notes": "Task graph execution stub; implement orchestration in future iteration.",
    }


def register_taskgraph_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="taskgraph.run",
            request_model=TaskGraphArgs,
            handler=_run_taskgraph,
            timeout_s=10.0,
        )
    )


__all__ = ["register_taskgraph_tools", "TaskGraphArgs", "TaskNode"]
