"""Strict public schemas for the local SRG-Tracker API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class WeeklyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt_id: str = Field(min_length=1, max_length=120)
    course_id: str = Field(min_length=1, max_length=40)
    semester: int = Field(ge=1, le=20)
    week: int = Field(ge=1, le=14)
    weekly_grade: float = Field(ge=0, le=100)
    attendance: float = Field(ge=0, le=100)
    assignment_score: float | None = Field(default=None, ge=0, le=100)
    quiz_score: float | None = Field(default=None, ge=0, le=100)
    submission_delay_days: float = Field(ge=0, le=30)
    corrections_count: int = Field(ge=0, le=100)
    assignment_completed: int = Field(ge=0, le=1)
    quiz_completed: int = Field(ge=0, le=1)
    midterm_score: float | None = Field(default=None, ge=0, le=100)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    history: list[WeeklyRecord] = Field(min_length=1, max_length=14)


class PredictionResponse(BaseModel):
    student_id: str
    course_id: str
    semester: int
    cutoff_week: int
    predicted_final_gpa: float
    predicted_course_outcome: str
    enroll_probability: float
    predicted_learning_pace: str
    advisory: str


class ModelInfo(BaseModel):
    task: str
    model_type: str
    validation_metric: str
    validation_value: float
    rank: int
    artifact: str
    artifact_available: bool
    selection_method: str


class HealthResponse(BaseModel):
    status: str
    models_ready: bool
    version: str = "4.0"
    registry_error: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)


class ErrorBody(BaseModel):
    code: str
    message: str
    fields: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
