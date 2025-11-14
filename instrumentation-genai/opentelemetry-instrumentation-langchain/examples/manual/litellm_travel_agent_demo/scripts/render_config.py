#!/usr/bin/env python3
"""Render a config template supporting ${VAR:-default} style placeholders.

Usage:
  ./scripts/render_config.py litellm-config.example.yaml litellm-config.yaml

The script replaces occurrences of ${NAME:-default} with the environment
variable NAME if set, otherwise with the provided default. It also replaces
${NAME} or $NAME with the environment value (empty string if unset).
"""

import os
import re
import sys


pattern = re.compile(r"\$\{([^:}]+)(:-([^}]*))?\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def replace_var(m):
    if m.group(1):
        name = m.group(1)
        default = m.group(3) if m.group(3) is not None else ""
        return os.environ.get(name, default)
    else:
        name = m.group(4)
        return os.environ.get(name, "")


def render(text: str) -> str:
    return pattern.sub(lambda m: replace_var(m), text)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) == 0:
        infile = "litellm-config.example.yaml"
        outfile = "litellm-config.yaml"
    elif len(argv) == 1:
        infile = argv[0]
        outfile = "litellm-config.yaml"
    else:
        infile, outfile = argv[0], argv[1]

    with open(infile, "r", encoding="utf-8") as f:
        src = f.read()

    dst = render(src)

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(dst)

    print(f"Rendered {infile} -> {outfile}")


if __name__ == "__main__":
    main()
