"""CLI entrypoint (placeholder).

The full ``demo | run | chat | markup`` subcommands are implemented in section-08 (api-ui). This
placeholder exists so the ``delta-chat`` console-script and ``python -m delta_chat.cli`` resolve from
section-01 onward instead of raising ModuleNotFoundError.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    print(
        "delta-chat CLI is not wired yet (implemented in section-08).\n"
        "Planned subcommands: demo | run | chat | markup.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
