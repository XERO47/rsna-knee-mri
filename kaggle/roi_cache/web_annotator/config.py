"""Runtime configuration for the remote ROI annotator.

Every value here is either derived from validated CLI/environment input or is
an explicitly labelled placeholder default. Placeholder defaults correspond to
"Decisions intentionally left for human sign-off" in
``references/REMOTE_ROI_ANNOTATION_APP_SPEC.md`` section 17; the application
never invents a stricter or looser value on its own, but it must start with
*something*, so the placeholder is documented here and in the operator
README rather than hidden inside argparse defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# PLACEHOLDER (spec section 17: "access-token lifetime and annotation-session
# dates"). The project owner must confirm or override this before the pilot.
DEFAULT_SESSION_LIFETIME_DAYS = 90

DEFAULT_MAX_BACKUPS = 20
DEFAULT_SLICE_CACHE_ITEMS = 64
DEFAULT_SLICE_CACHE_BYTES = 200 * 1024 * 1024  # 200 MiB bounded PNG cache
DEFAULT_MIN_FREE_DISK_GB = 5.0
DEFAULT_MAX_CONCURRENT_DECODES = 2

# Server-side tolerance for the pixel -> patient -> pixel round trip, matching
# the desktop annotator's acceptance gate in annotate_roi_pilot.py.
ROUND_TRIP_TOLERANCE_PX = 1e-5

PILOT_STUDY_COUNT = 20
FINAL_STUDY_COUNT = 100

UNABLE_REASONS = (
    "anatomy_not_visible",
    "reference_series_unsuitable",
    "corrupt_unreadable_image",
    "laterality_concern",
    "other",
)


@dataclass(frozen=True)
class AppConfig:
    competition_root: Path
    work_dir: Path
    db_path: Path
    backup_dir: Path
    log_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    session_lifetime_days: int = DEFAULT_SESSION_LIFETIME_DAYS
    max_backups: int = DEFAULT_MAX_BACKUPS
    slice_cache_items: int = DEFAULT_SLICE_CACHE_ITEMS
    slice_cache_bytes: int = DEFAULT_SLICE_CACHE_BYTES
    min_free_disk_gb: float = DEFAULT_MIN_FREE_DISK_GB
    max_concurrent_decodes: int = DEFAULT_MAX_CONCURRENT_DECODES

    @classmethod
    def from_work_dir(
        cls,
        competition_root: str | Path,
        work_dir: str | Path,
        **overrides: object,
    ) -> "AppConfig":
        work_dir_path = Path(work_dir)
        return cls(
            competition_root=Path(competition_root),
            work_dir=work_dir_path,
            db_path=work_dir_path / "annotator.sqlite3",
            backup_dir=work_dir_path / "backups",
            log_dir=work_dir_path / "logs",
            **overrides,  # type: ignore[arg-type]
        )

    def ensure_directories(self) -> None:
        for directory in (self.work_dir, self.backup_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
