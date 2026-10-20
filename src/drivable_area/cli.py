"""Small command dispatcher; heavy model imports remain optional."""

import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="도로/차로 중심 추정 및 방향 표시")
    parser.add_argument("command", choices=["video", "arrows"])
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    if args.command == "video":
        from .pipeline import main as run
    else:
        from .replay import main as run
    return run(remaining)
