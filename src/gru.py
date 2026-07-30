"""Masked raw-sequence dataset and shared multi-task GRU implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, mean_squared_error
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    COURSE_WEEKS,
    EVALUATION_CUTOFFS,
    RANDOM_STATE,
    RAW_WEEKLY_FEATURE_COLUMNS,
)
from src.features import build_sequence
from src.preprocessing import history_from_group


PACE_TO_INDEX = {"behind": 0, "on_track": 1, "ahead": 2}
INDEX_TO_PACE = {value: key for key, value in PACE_TO_INDEX.items()}


@dataclass
class SequenceEncoder:
    """Train-fitted numeric scaling and course encoding state."""

    course_to_index: dict[str, int]
    means: list[float]
    scales: list[float]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "SequenceEncoder":
        courses = sorted(str(value) for value in frame["course_id"].unique())
        rows: list[list[float]] = []
        for _, group in frame.groupby("attempt_id", sort=False):
            history = history_from_group(group, COURSE_WEEKS)
            values, mask = build_sequence(history, max_weeks=COURSE_WEEKS)
            rows.extend(values[mask].tolist())
        matrix = np.asarray(rows, dtype=np.float32)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales < 1e-6] = 1.0
        return cls(
            course_to_index={course: index for index, course in enumerate(courses)},
            means=means.tolist(),
            scales=scales.tolist(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SequenceEncoder":
        return cls(
            course_to_index={str(k): int(v) for k, v in payload["course_to_index"].items()},
            means=[float(value) for value in payload["means"]],
            scales=[float(value) for value in payload["scales"]],
        )

    @property
    def input_size(self) -> int:
        return len(self.means) + len(self.course_to_index) + 1

    def transform_history(
        self,
        history: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        values, mask = build_sequence(history, max_weeks=COURSE_WEEKS)
        scaled = (values - np.asarray(self.means, dtype=np.float32)) / np.asarray(
            self.scales,
            dtype=np.float32,
        )
        scaled[~mask] = 0.0
        context = np.zeros(
            (COURSE_WEEKS, len(self.course_to_index) + 1),
            dtype=np.float32,
        )
        course = str(history[0]["course_id"])
        if course in self.course_to_index:
            context[: len(history), self.course_to_index[course]] = 1.0
        context[: len(history), -1] = (float(history[0]["semester"]) - 1.5) / 0.5
        return np.concatenate([scaled, context], axis=1), mask


@dataclass
class SequenceDataset:
    values: np.ndarray
    masks: np.ndarray
    gpa: np.ndarray
    outcome: np.ndarray
    pace: np.ndarray
    cutoffs: np.ndarray
    metadata: pd.DataFrame


def build_sequence_dataset(
    frame: pd.DataFrame,
    encoder: SequenceEncoder,
    cutoffs: Iterable[int] = EVALUATION_CUTOFFS,
) -> SequenceDataset:
    """Build one padded raw sequence per attempt and cutoff."""
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    gpa: list[float] = []
    outcome: list[int] = []
    pace: list[int] = []
    cutoff_values: list[int] = []
    metadata: list[dict[str, object]] = []
    for _, group in frame.groupby("attempt_id", sort=False):
        first = group.iloc[0]
        for cutoff in cutoffs:
            history = history_from_group(group, int(cutoff))
            sequence, mask = encoder.transform_history(history)
            values.append(sequence)
            masks.append(mask)
            gpa.append(float(first["final_gpa"]))
            outcome.append(int(first["course_outcome"] == "enroll"))
            pace.append(PACE_TO_INDEX[str(first["learning_pace"])])
            cutoff_values.append(int(cutoff))
            metadata.append(
                {
                    "student_id": first["student_id"],
                    "attempt_id": first["attempt_id"],
                    "course_id": first["course_id"],
                    "semester": int(first["semester"]),
                    "cutoff_week": int(cutoff),
                }
            )
    return SequenceDataset(
        values=np.asarray(values, dtype=np.float32),
        masks=np.asarray(masks, dtype=bool),
        gpa=np.asarray(gpa, dtype=np.float32),
        outcome=np.asarray(outcome, dtype=np.float32),
        pace=np.asarray(pace, dtype=np.int64),
        cutoffs=np.asarray(cutoff_values, dtype=np.int64),
        metadata=pd.DataFrame(metadata),
    )


class MultiTaskGRU(nn.Module):
    """Shared GRU encoder with GPA, outcome, and pace heads."""

    def __init__(self, input_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
        )
        self.gpa_head = nn.Linear(32, 1)
        self.outcome_head = nn.Linear(32, 1)
        self.pace_head = nn.Linear(32, 3)

    def forward(
        self,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lengths = masks.sum(dim=1).to(torch.int64).cpu()
        packed = pack_padded_sequence(
            values,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        representation = self.shared(hidden[-1])
        gpa = torch.clamp(self.gpa_head(representation).squeeze(1), 0.0, 4.5)
        outcome = self.outcome_head(representation).squeeze(1)
        pace = self.pace_head(representation)
        return gpa, outcome, pace

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _loader(dataset: SequenceDataset, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(dataset.values),
        torch.from_numpy(dataset.masks),
        torch.from_numpy(dataset.gpa),
        torch.from_numpy(dataset.outcome),
        torch.from_numpy(dataset.pace),
    )
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict_gru(
    model: MultiTaskGRU,
    dataset: SequenceDataset,
    *,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Return decoded predictions and probabilities."""
    model.eval()
    gpa_values: list[np.ndarray] = []
    outcome_probability: list[np.ndarray] = []
    pace_values: list[np.ndarray] = []
    pace_probability: list[np.ndarray] = []
    with torch.no_grad():
        for values, masks, *_ in _loader(dataset, batch_size, False):
            gpa, outcome_logits, pace_logits = model(values, masks)
            gpa_values.append(gpa.numpy())
            outcome_probability.append(torch.sigmoid(outcome_logits).numpy())
            pace_values.append(torch.argmax(pace_logits, dim=1).numpy())
            pace_probability.append(torch.softmax(pace_logits, dim=1).numpy())
    probability = np.concatenate(outcome_probability)
    return {
        "gpa": np.concatenate(gpa_values),
        "outcome_probability": probability,
        "outcome": np.where(probability >= 0.5, "enroll", "pass"),
        "pace": np.asarray(
            [INDEX_TO_PACE[int(index)] for index in np.concatenate(pace_values)]
        ),
        "pace_probability": np.concatenate(pace_probability),
    }


def _validation_objective(
    model: MultiTaskGRU,
    dataset: SequenceDataset,
) -> float:
    predictions = predict_gru(model, dataset)
    rmse = mean_squared_error(dataset.gpa, predictions["gpa"]) ** 0.5
    outcome_f1 = f1_score(
        np.where(dataset.outcome == 1, "enroll", "pass"),
        predictions["outcome"],
        pos_label="enroll",
        zero_division=0,
    )
    pace_f1 = f1_score(
        np.asarray([INDEX_TO_PACE[int(index)] for index in dataset.pace]),
        predictions["pace"],
        average="macro",
        zero_division=0,
    )
    return float(rmse + (1.0 - outcome_f1) + (1.0 - pace_f1))


def train_gru(
    train: SequenceDataset,
    validation: SequenceDataset,
    input_size: int,
    *,
    max_epochs: int = 24,
    patience: int = 5,
    batch_size: int = 256,
) -> tuple[MultiTaskGRU, list[dict[str, float]], float]:
    """Train with Adam and restore the best validation checkpoint."""
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.use_deterministic_algorithms(True)
    model = MultiTaskGRU(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    gpa_loss = nn.MSELoss()
    outcome_loss = nn.BCEWithLogitsLoss()
    pace_loss = nn.CrossEntropyLoss()
    best_state: dict[str, torch.Tensor] | None = None
    best_objective = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses: list[float] = []
        for values, masks, gpa, outcome, pace in _loader(train, batch_size, True):
            optimizer.zero_grad()
            predicted_gpa, outcome_logits, pace_logits = model(values, masks)
            loss = (
                gpa_loss(predicted_gpa, gpa)
                + 0.8 * outcome_loss(outcome_logits, outcome)
                + 0.8 * pace_loss(pace_logits, pace)
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        objective = _validation_objective(model, validation)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_objective": objective,
            }
        )
        if objective < best_objective - 1e-4:
            best_objective = objective
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    training_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("GRU training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, history, float(training_seconds)
