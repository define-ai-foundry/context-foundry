#!/usr/bin/env python3
"""The steps of the fusion stage's publish that pipeline-stages' publish_descriptor.py cannot
take for an image outside `pipeline-stages/`; everything else is that script, from a checkout.

    python3 stage/publish.py decide --catalog <checkout of the catalog's main>
    python3 stage/publish.py fill --stages <pipeline-stages checkout> --digest <d> \\
        --version <v> --catalog <checkout of the catalog's main> --staging-dir <dir>
    python3 stage/publish.py retarget-body <pr-body.md>

`decide` publishes only when `[project].version` ranks above the version the catalog's main
pins for the stage type, or the catalog has none. `decide` and `fill` print `key=value`
lines for `$GITHUB_OUTPUT`.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import tomllib
import yaml

STAGE = "fusion"
REPOSITORY = "context-foundry/fusion"
SOURCE = Path(__file__).parent / STAGE
WORKFLOW = ".github/workflows/stage.yaml"
PLACEHOLDER_DIGEST = "sha256:" + "0" * 64


def load(path: Path) -> ModuleType:
    """Import the script at `path` as a module."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def emit(values: dict[str, str]) -> None:
    for key, value in values.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"the {key} output spans more than one line")
        print(f"{key}={value}")


def decide(catalog: Path, pyproject: Path) -> dict[str, str]:
    """Whether to publish `[project].version`, against what the catalog's main pins."""
    precedence = load(catalog / "scripts" / "check_versions.py").precedence
    version = tomllib.loads(pyproject.read_text())["project"]["version"]
    if precedence(version) is None:
        raise ValueError(f"[project].version {version!r} is not a semantic version")
    path = catalog / "stages" / STAGE / "descriptor.yaml"
    image = {}
    if path.is_file():
        image = (load_yaml(path.read_text()) or {}).get("image") or {}
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


def load_yaml(text: str) -> dict | None:
    document = yaml.safe_load(text)
    return document if isinstance(document, dict) else None


def fill(stages: Path, catalog: Path, staging: Path, digest: str, version: str) -> dict[str, str]:
    """publish_descriptor.py fill for a build, against this repository's image repository."""
    p = load(stages / "scripts" / "publish_descriptor.py")
    staged = staging / STAGE
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(SOURCE, staged)
    text = (SOURCE / "descriptor.yaml").read_text()
    own = p.load(text)
    if own is None:
        raise ValueError(f"{SOURCE}/descriptor.yaml is not a YAML mapping")
    repository = (own.get("image") or {}).get("repository")
    if repository != REPOSITORY:
        raise ValueError(f"descriptor repository is {repository}, expected {REPOSITORY}")
    current = p.catalog_descriptor(catalog, STAGE)
    runtime = p.proposed_runtime(current, digest)
    rendered, dropped = p.render(text, digest, version, runtime)
    (staged / "descriptor.yaml").write_text(rendered)
    return {
        "comments_dropped": "true" if dropped else "false",
        "publisher": str(own.get("publisher")),
        "runtime": runtime,
        "own_runtime": str(own.get("runtime", "")),
        "catalog_runtime": str((current or {}).get("runtime", "")),
    }


def retarget_body(path: Path) -> None:
    """Name this repository's image repository and workflow in publish_descriptor.py's body."""
    text = path.read_text()
    for old, new in (
        (f"**Repository**: pipeline-stages/{STAGE}", f"**Repository**: {REPOSITORY}"),
        ("`.github/workflows/publish.yaml`", f"`{WORKFLOW}`"),
    ):
        if text.count(old) != 1:
            raise ValueError(f"the body does not name {old!r} exactly once")
        text = text.replace(old, new)
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decide_cmd = commands.add_parser("decide", help="whether this version is to be published")
    decide_cmd.add_argument("--catalog", required=True, help="checkout of the catalog's main")
    decide_cmd.add_argument("--pyproject", default="pyproject.toml")
    fill_cmd = commands.add_parser("fill", help="fill the descriptor into a staging copy")
    fill_cmd.add_argument("--stages", required=True, help="checkout of pipeline-stages")
    fill_cmd.add_argument("--catalog", required=True, help="checkout of the catalog's main")
    fill_cmd.add_argument("--staging-dir", required=True)
    fill_cmd.add_argument("--digest", required=True)
    fill_cmd.add_argument("--version", required=True)
    body_cmd = commands.add_parser("retarget-body", help="rewrite the pull request body")
    body_cmd.add_argument("body")
    args = parser.parse_args(argv)

    try:
        if args.command == "decide":
            emit(decide(Path(args.catalog), Path(args.pyproject)))
        elif args.command == "fill":
            emit(
                fill(
                    Path(args.stages),
                    Path(args.catalog),
                    Path(args.staging_dir),
                    args.digest,
                    args.version,
                )
            )
        else:
            retarget_body(Path(args.body))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
