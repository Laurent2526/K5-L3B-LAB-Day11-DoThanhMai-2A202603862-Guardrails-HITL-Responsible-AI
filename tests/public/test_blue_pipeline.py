"""Verify the Blue pipeline with a local model stub and temporary artifacts."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema

ROOT = Path(__file__).resolve().parents[2]


def test_blue_suite_applies_guardrails_and_writes_artifacts(tmp_path, monkeypatch):
    from assignment import pipeline
    from core.openai_runtime import OpenAIRunner

    model_calls = []

    def respond(*, model, messages, temperature):
        model_calls.append((model, messages[-1]["content"]))
        reply = "A transfer may remain pending while the bank processes it."
        if messages[-1]["content"].startswith("Summarize this external email"):
            reply += " Customer email: example@example.invalid."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
    monkeypatch.setattr(OpenAIRunner, "_client", lambda self: client)
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    audit, monitor = pipeline.build_observability()
    plugins = pipeline.build_production_plugins()

    result = asyncio.run(pipeline.run_assignment_suite({
        "audit": audit, "monitor": monitor, "plugins": plugins,
    }))

    schema = json.loads((ROOT / "schemas" / "results.schema.json").read_text(encoding="utf-8"))
    on_disk = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert on_disk == result
    jsonschema.validate(on_disk, schema)
    assert all(not query["blocked"] for query in result["safe_queries"])
    assert all(query["layer"] == "input_guardrail" for query in result["attack_queries"])
    assert result["rate_limit"]["passed"] == 10
    assert result["rate_limit"]["blocked"] == 5
    assert len(model_calls) == 6
    assert {model for model, _ in model_calls} == {"liquid/lfm-3-1.7b:free"}
    email_case = result["edge_cases"][2]
    assert email_case["layer"] == "output_guardrail"
    assert "[REDACTED]" in email_case["response_preview"]
    assert "example@example.invalid" not in email_case["response_preview"]
    assert result["egress_checks"] == {
        "trusted_banking_payload": True,
        "sensitive_payload": False,
        "untrusted_destination": False,
    }

    logs = json.loads((tmp_path / "audit_log.json").read_text(encoding="utf-8"))
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["total_requests"] == len(logs) == 31
    assert metrics["rate_limit_hits"] == 5
    assert len({entry["request_id"] for entry in logs}) == len(logs)
    assert all(entry["latency_ms"] >= 0 for entry in logs)
    assert "example@example.invalid" not in json.dumps(logs)
