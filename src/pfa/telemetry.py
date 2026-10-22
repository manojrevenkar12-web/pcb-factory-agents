"""OpenTelemetry tracing: every judge call and tool call is a span with its cost and outcome.

Spans go to an OTLP collector when OTEL_EXPORTER_OTLP_ENDPOINT is set (with the optional
exporter installed), and nowhere otherwise. Tests attach an in-memory exporter.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

_provider: TracerProvider | None = None


def provider() -> TracerProvider:
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": "pcb-factory-agents"}))
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            except ImportError:  # exporter not installed: tracing stays local
                pass
    return _provider


def tracer(name: str = "pfa") -> trace.Tracer:
    return provider().get_tracer(name)


def add_exporter(exporter: SpanExporter) -> None:
    provider().add_span_processor(SimpleSpanProcessor(exporter))
