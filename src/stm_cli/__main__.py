#!/usr/bin/env python3
"""Unified STM CLI entrypoint."""

from __future__ import annotations

import sys

from sep_text_manifold.cli_plots import main as plots_main
from sep_text_manifold.cli_lead import main as lead_main
from sep_text_manifold import cli as legacy_cli

# Placeholder imports for future subcommands
try:
    from stm_stream.core import main as stream_main  # type: ignore
except Exception:  # pragma: no cover - placeholder
    stream_main = None

try:
    from stm_onsets.cli import main as onsets_main  # type: ignore
except Exception:  # pragma: no cover - placeholder
    onsets_main = None


def main() -> None:
    if len(sys.argv) <= 1:
        legacy_cli.main()
        return
    first = sys.argv[1]
    if first not in {"plots", "lead", "stream", "onsets"}:
        legacy_cli.main()
        return
    command = sys.argv[1]
    if command == "plots":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        plots_main()
    elif command == "lead":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        lead_main()
    elif command == "stream":
        if stream_main is None:
            print("stream module not yet implemented", file=sys.stderr)
            sys.exit(1)
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        stream_main()
    elif command == "onsets":
        if onsets_main is None:
            print("onsets module not yet implemented", file=sys.stderr)
            sys.exit(1)
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        onsets_main()



if __name__ == "__main__":
    main()
