from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from jarvis.telemetry.logging import get_logger

_configured = False


def configure_tracing(service_name: str, endpoint: str | None) -> None:
    global _configured
    if _configured or endpoint is None:
        return

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    get_logger(__name__).info("tracing.enabled", endpoint=endpoint, service_name=service_name)
    _configured = True

