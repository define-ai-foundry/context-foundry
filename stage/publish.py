#!/usr/bin/env python3
"""Whether the fusion stage's `[project].version` is to be published to the catalog.

    python3 stage/publish.py decide --catalog <checkout of the catalog's main>

It is, when the version ranks above the version the catalog's main pins for the stage type,
by semantic version precedence, or the catalog has none. Prints `key=value` lines for
`$GITHUB_OUTPUT`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib
import yaml

STAGE = "fusion"
REPOSITORY = "context-foundry/fusion"
PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
# The catalog schema's image.version pattern.
VERSION = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?"
)


def precedence(version: object) -> tuple | None:
    """The semantic version precedence of `version`, or None when it is not one."""
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        return None
    core, _, pre = version.partition("-")
    release = tuple(int(part) for part in core.split("."))
    if not pre:
        return (*release, (1,))
    # Numeric identifiers rank below alphanumeric ones; a longer list ranks above its prefix.
    ids = tuple((0, int(i), "") if i.isdigit() else (1, 0, i) for i in pre.split("."))
    return (*release, (0, ids))


def emit(values: dict[str, str]) -> None:
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"the {key} output spans more than one line")
        print(f"{key}={value}")


def decide(catalog: Path, pyproject: Path) -> dict[str, str]:
    """Whether to publish `[project].version`, against what the catalog's main pins."""
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    if precedence(version) is None:
        raise ValueError(f"[project].version {version!r} is not a semantic version")
    path = catalog / "stages" / STAGE / "descriptor.yaml"
    image = {}
    if path.is_file():
        document = yaml.safe_load(path.read_text())
        image = (document if isinstance(document, dict) else {}).get("image") or {}
    if image.get("repository") not in (None, REPOSITORY):
        raise ValueError(f"the catalog's {STAGE} pins {image.get('repository')}, not {REPOSITORY}")
    pinned = image.get("version")
    if not image or image.get("digest") == PLACEHOLDER_DIGEST:
        publish, reason = True, f"the catalog has no published {STAGE} image"
    elif precedence(pinned) is None:
        raise ValueError(f"the catalog's {STAGE} version {pinned!r} is not a semantic version")
    elif precedence(version) > precedence(pinned):
        publish, reason = True, f"{version} ranks above the catalog's {pinned}"
    else:
        publish = False
        reason = f"the catalog already pins {pinned}; raise [project].version above it to publish"
    return {
        "publish": "true" if publish else "false",
        "version": version,
        "pinned": str(pinned or ""),
        "reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decide_cmd = commands.add_parser("decide", help="whether this version is to be published")
    decide_cmd.add_argument("--catalog", required=True, help="checkout of the catalog's main")
    decide_cmd.add_argument("--pyproject", default="pyproject.toml")
    args = parser.parse_args(argv)
    try:
        emit(decide(Path(args.catalog), Path(args.pyproject)))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
