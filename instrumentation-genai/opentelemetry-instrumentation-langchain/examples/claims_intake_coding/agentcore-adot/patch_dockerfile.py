#!/usr/bin/env python3
"""Bump the ADOT version the AgentCore starter toolkit's Dockerfile template hardcodes.

Must run BETWEEN `agentcore configure` and `agentcore launch`, because configure
regenerates the file from a template every time and silently discards any manual edit --
the same trap documented in ai_underwriting_pipeline/agentcore-splunk/patch_dockerfile.py.

The template (bedrock_agentcore_starter_toolkit/utils/runtime/templates/Dockerfile.j2) pins

    RUN uv pip install aws-opentelemetry-distro==0.12.2

which resolves to opentelemetry-sdk==1.33.1 / opentelemetry-instrumentation==0.54b1 --
whose semantic-conventions generation predates `GEN_AI_CONVERSATION_ID`. That is the actual
root cause behind "ADOT breaks GenAI instrumentation": not that ADOT is incompatible with
GenAI semconv in general, but that the starter toolkit template is frozen on a year-old ADOT
release. The current release (0.19.0) resolves to opentelemetry-sdk==1.44.0 /
opentelemetry-semantic-conventions==0.65b0, which has the symbol -- verified directly against
the wheel rather than assumed, since PyPI metadata alone does not resolve transitive pins.

This agent does not use the Splunk GenAI instrumentation packages (see requirements.txt), so
unlike the reordering patch in the sibling `agentcore-splunk` example, no install-order fix is
needed here: nothing else in requirements.txt pins the OTel core tightly enough to conflict
with ADOT's own pin.
"""
import re
import sys
from pathlib import Path

ADOT_VERSION = "0.19.0"

GUARD = f'''
# Fail the build rather than ship an image whose GenAI instrumentation cannot import.
RUN python -c "\\
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as g; \\
assert hasattr(g, 'GEN_AI_CONVERSATION_ID'), 'semconv too old: aws-opentelemetry-distro=={ADOT_VERSION} pin regressed'; \\
from opentelemetry.instrumentation.openai import OpenAIInstrumentor; \\
print('instrumentation import OK')"
'''


_DISTRO_LINE = re.compile(r"^RUN uv pip install aws-opentelemetry-distro==\S+$")


def patch(path: Path) -> bool:
    text = path.read_text()
    if "instrumentation import OK" in text:
        print(f"  already patched: {path}")
        return True

    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if _DISTRO_LINE.match(line.rstrip("\n")):
            lines[i] = f"RUN uv pip install aws-opentelemetry-distro=={ADOT_VERSION}\n"
            lines.insert(i + 1, GUARD.lstrip("\n") if GUARD.startswith("\n") else GUARD)
            path.write_text("".join(lines))
            print(f"  patched: {path} (aws-opentelemetry-distro -> {ADOT_VERSION})")
            return True

    print(f"  ERROR: no aws-opentelemetry-distro install line found in {path}", file=sys.stderr)
    print("  (was --disable-otel passed to `agentcore configure`? this agent needs it NOT set)", file=sys.stderr)
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: patch_dockerfile.py <agent-name> [<agent-name> ...]")
    ok = True
    for agent in sys.argv[1:]:
        p = Path(".bedrock_agentcore") / agent / "Dockerfile"
        if not p.exists():
            print(f"  ERROR: {p} does not exist (run agentcore configure first)", file=sys.stderr)
            ok = False
            continue
        ok = patch(p) and ok
    sys.exit(0 if ok else 1)
