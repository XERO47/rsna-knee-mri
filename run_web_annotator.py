"""Start the ROI web annotator with one shared account and all patients.

From the repository root, run only::

    python run_web_annotator.py

The first run initializes the pilot; later runs resume it without modifying
saved annotations. Advanced path/phase options are available via ``--help``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from types import SimpleNamespace

from kaggle.roi_cache.web_annotator import cli
from kaggle.roi_cache.web_annotator.config import AppConfig
from kaggle.roi_cache.web_annotator.single_user import prepare_single_user


REPOSITORY_ROOT = Path(__file__).resolve().parent


def _open_when_ready(url: str, host: str, port: int) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.25)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--competition-root", default=str(REPOSITORY_ROOT.parent / "data-root"))
    parser.add_argument("--work-dir", default=str(REPOSITORY_ROOT / "roi_work" / "web_annotator_single"))
    parser.add_argument("--cohort", default=None, help="Frozen cohort name; defaults to the selected phase.")
    parser.add_argument("--manifest", default=None, help="Explicit frozen manifest path; must be paired with --template.")
    parser.add_argument("--template", default=None, help="Explicit frozen annotation template; must be paired with --manifest.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    phase = arguments.phase
    if bool(arguments.manifest) != bool(arguments.template):
        raise ValueError("--manifest and --template must be provided together")
    if arguments.manifest:
        manifest = Path(arguments.manifest)
        template = Path(arguments.template)
        cohort = arguments.cohort or f"{phase}-custom"
    elif phase == "pilot":
        cohort = "pilot"
        manifest = REPOSITORY_ROOT / "roi_work" / "pilot" / "pilot_manifest.csv"
        template = REPOSITORY_ROOT / "roi_work" / "pilot" / "pilot_annotations.csv"
    else:
        cohort = "final"
        manifest = REPOSITORY_ROOT / "roi_work" / "annotations" / "annotation_manifest.csv"
        template = REPOSITORY_ROOT / "roi_work" / "annotations" / "roi_annotations.csv"
    if arguments.cohort and not arguments.manifest:
        cohort = arguments.cohort

    config = AppConfig.from_work_dir(arguments.competition_root, arguments.work_dir)
    setup = prepare_single_user(
        config=config,
        cohort_name=cohort,
        phase=phase,
        manifest_path=manifest,
        template_path=template,
        base_url=f"http://{arguments.host}:{arguments.port}",
    )

    print("\nROI annotator ready", flush=True)
    print(f"  Shared account: {setup.reader_name}", flush=True)
    print(f"  Patients: {setup.study_count}", flush=True)
    print(f"  Private login link: {setup.login_url}", flush=True)
    print(f"  Private token backup: {setup.token_path}", flush=True)
    print("  Every patient is already available; no assignment step is needed.\n", flush=True)

    common = {
        "work_dir": str(config.work_dir),
        "competition_root": str(config.competition_root),
        "cohort": cohort,
    }
    cli.command_preflight(SimpleNamespace(**common))

    if not arguments.no_open_browser:
        threading.Thread(
            target=_open_when_ready,
            args=(setup.login_url, arguments.host, arguments.port),
            daemon=True,
        ).start()

    caffeinate = None
    if shutil.which("caffeinate"):
        caffeinate = subprocess.Popen(
            ["caffeinate", "-dis", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    try:
        cli.command_serve(SimpleNamespace(
            **common,
            host=arguments.host,
            port=arguments.port,
        ))
    finally:
        if caffeinate is not None and caffeinate.poll() is None:
            caffeinate.terminate()


if __name__ == "__main__":
    main()
