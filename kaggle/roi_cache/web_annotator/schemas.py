"""Strict, bounded request/response validation for the HTTP API."""

from __future__ import annotations

import math
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import UNABLE_REASONS

MAX_STRING_LENGTH = 4096


def _validate_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


class SaveAnnotationRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=512)
    sop_uid: str = Field(min_length=1, max_length=256)
    row_px: float
    column_px: float
    expected_revision: int = Field(ge=0)
    request_id: str = Field(min_length=1, max_length=64)

    @field_validator("row_px", "column_px")
    @classmethod
    def _finite_pixel(cls, value: float) -> float:
        return _validate_finite(value)

    @field_validator("request_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("request_id must be a UUID string") from error
        return value


class UnableRequest(BaseModel):
    task_id: str = Field(min_length=1, max_length=512)
    reason: Literal[UNABLE_REASONS]  # type: ignore[valid-type]
    notes: str | None = Field(default=None, max_length=MAX_STRING_LENGTH)
    expected_revision: int = Field(ge=0)
    request_id: str = Field(min_length=1, max_length=64)

    @field_validator("request_id")
    @classmethod
    def _valid_uuid(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except ValueError as error:
            raise ValueError("request_id must be a UUID string") from error
        return value
