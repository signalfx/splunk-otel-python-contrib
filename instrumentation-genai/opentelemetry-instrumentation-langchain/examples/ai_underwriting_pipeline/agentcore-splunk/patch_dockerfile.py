#!/usr/bin/env python3
"""Reorder the Dockerfile that `agentcore configure` generates.

Must run BETWEEN `agentcore configure` and `agentcore launch`, because configure
regenerates the file from a template every time and silently discards any manual edit.
Learning this the hard way costs a full build/deploy/invoke cycle: the rebuilt image looks
fine, deploys green, and then dies at import with the same error as before.

What it changes: the template installs `aws-opentelemetry-distro` AFTER the application
requirements. That distro `==`-pins the whole OpenTelemetry stack, so installing it last
downgrades opentelemetry-semantic-conventions beneath what the Splunk GenAI instrumentation
needs, and the container dies at import with

    AttributeError: module '...gen_ai_attributes' has no attribute 'GEN_AI_CONVERSATION_ID'

which surfaces to the operator only as "Runtime health check failed or timed out".
Installing the distro first lets the application requirements win the resolution.

It also appends an import guard so a downgraded stack fails the BUILD instead of shipping
an image that cannot start.
"""
import re
import sys
from pathlib import Path

GUARD = '''
# Fail the build rather than ship an image whose instrumentation cannot import.
RUN python -c "\\
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as g; \\
assert hasattr(g, 'GEN_AI_CONVERSATION_ID'), 'semconv too old: the OTel stack was downgraded'; \\
from opentelemetry.instrumentation.langchain import LangchainInstrumentor; \\
print('instrumentation import OK')"
'''


def patch(path: Path) -> bool:
    t = path.read_text()
    if "instrumentation import OK" in t:
        print(f"  already patched: {path}")
        return True

    m_req = re.search(r"^RUN uv pip install -r requirements\.txt\s*$", t, re.M)
    m_distro = re.search(r"^RUN uv pip install (aws-opentelemetry-distro\S*)\s*$", t, re.M)
    if not m_req:
        print(f"  ERROR: requirements install line not found in {path}", file=sys.stderr)
        return False
    if not m_distro:
        # Expected when configure was run with --disable-otel: there is no distro to
        # reorder, so only the import guard is added.
        print("  no aws-opentelemetry-distro present (--disable-otel); adding guard only")
    elif m_distro.start() < m_req.start():
        print("  distro already installed first; adding guard only")
    else:
        # Remove the trailing distro line and re-insert it before the requirements install.
        t = t[: m_distro.start()] + t[m_distro.end():]
        m_req = re.search(r"^RUN uv pip install -r requirements\.txt\s*$", t, re.M)
        reordered = (
            "# Layer order is load-bearing: the distro pins the OTel stack with ==, so it must\n"
            "# be installed BEFORE the app requirements or it downgrades semantic-conventions\n"
            "# beneath the GenAI instrumentation. Patched by patch_dockerfile.py.\n"
            f"RUN uv pip install {m_distro.group(1)}\n"
            "RUN uv pip install -r requirements.txt\n"
        )
        t = t[: m_req.start()] + reordered + t[m_req.end():]

    # Insert the guard right after the install block, before USER drops privileges.
    anchor = re.search(r"^RUN uv pip install -r requirements\.txt\s*$", t, re.M)
    t = t[: anchor.end()] + "\n" + GUARD + t[anchor.end():]
    path.write_text(t)
    print(f"  patched: {path}")
    return True


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
