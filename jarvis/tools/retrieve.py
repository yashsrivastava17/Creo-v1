from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from jarvis.tools.registry import ToolContext, ToolRegistry, ToolSpec


class RetrieveTopKArgs(BaseModel):
    q: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)


class SummarizeContextArgs(BaseModel):
    snippets: list[str] = Field(default_factory=list)
    max_tokens: int = Field(400, ge=50, le=1200)


async def _retrieve_topk(args: RetrieveTopKArgs, ctx: ToolContext) -> dict[str, Any]:
    if ctx.memory is None or not hasattr(ctx.memory, "search"):
        return {"hits": [], "tokens_estimate": 0}
    hits = await ctx.memory.search(args.q, limit=args.k, filters=args.filters)
    tokens_estimate = sum(len(hit.get("snippet", "")) // 4 + 1 for hit in hits)
    return {"hits": hits, "tokens_estimate": tokens_estimate}


async def _summarize_context(args: SummarizeContextArgs, _: ToolContext) -> dict[str, Any]:
    if not args.snippets:
        return {"summary": "", "sources": []}
    combined = " ".join(args.snippets)
    max_chars = args.max_tokens * 4
    summary = combined[:max_chars]
    if len(combined) > max_chars:
        summary = summary.rsplit(" ", 1)[0] + "..."
    return {"summary": summary.strip(), "sources": list(range(len(args.snippets)))}


def register_retrieval_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="retrieve.topk",
            request_model=RetrieveTopKArgs,
            handler=_retrieve_topk,
            timeout_s=6.0,
        )
    )
    registry.register(
        ToolSpec(
            name="summarize.context",
            request_model=SummarizeContextArgs,
            handler=_summarize_context,
            timeout_s=4.0,
        )
    )


__all__ = ["register_retrieval_tools", "RetrieveTopKArgs", "SummarizeContextArgs"]
