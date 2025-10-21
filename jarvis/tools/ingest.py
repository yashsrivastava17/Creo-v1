from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator

from jarvis.tools.registry import ToolContext, ToolRegistry, ToolSpec


class IngestBatchArgs(BaseModel):
    paths: list[str] = Field(default_factory=list, description="Filesystem paths to files or directories")
    dry_run: bool = Field(False, description="If true, do not persist; just report.")

    @validator("paths", each_item=True)
    def path_not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("path cannot be empty")
        return value


async def _ingest_batch(args: IngestBatchArgs, ctx: ToolContext) -> dict[str, Any]:
    count_files = 0
    missing: list[str] = []
    for p in args.paths:
        path = Path(p).expanduser()
        if path.exists():
            if path.is_file():
                count_files += 1
            else:
                count_files += sum(1 for _ in path.rglob("*") if _.is_file())
        else:
            missing.append(str(path))
    result = {
        "ingested": 0 if args.dry_run else count_files,
        "discovered": count_files,
        "missing": missing,
        "dry_run": args.dry_run,
        "tokens_estimate": count_files * 800,
    }
    ctx.extras["last_ingest"] = result
    return result


def register_ingest_tools(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="ingest.batch",
            request_model=IngestBatchArgs,
            handler=_ingest_batch,
            timeout_s=30.0,
        )
    )


__all__ = ["register_ingest_tools", "IngestBatchArgs"]
