#!/usr/bin/env python3
"""Unified STM CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from sep_text_manifold.cli_plots import main as plots_main
from sep_text_manifold.cli_lead import main as lead_main

# Placeholder imports for future subcommands
try:
    from stm_stream.core import main as stream_main  # type: ignore
except Exception:  # pragma: no cover - placeholder
    stream_main = None

try:
    from stm_onsets.cli import main as onsets_main  # type: ignore
except Exception:  # pragma: no cover - placeholder
    onsets_main = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="stm", description="STM toolkit CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("plots", help="Render structural overlay plots")
    sub.add_parser("lead", help="Compute lead-time foreground density")
    sub.add_parser("stream", help="Launch streaming router")
    sub.add_parser("onsets", help="Auto-label onset times")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plots":
        plots_main()
    elif args.command == "lead":
        lead_main()
    elif args.command == "stream":
        if stream_main is None:
            print("stream module not yet implemented", file=sys.stderr)
            sys.exit(1)
        stream_main()
    elif args.command == "onsets":
        if onsets_main is None:
            print("onsets module not yet implemented", file=sys.stderr)
            sys.exit(1)
        onsets_main()
    else:
        print("Available commands: plots, lead, stream, onsets", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
