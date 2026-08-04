import type { Health, Identity, ModelInfo, Prediction, WeekDraft } from "./types";

function numberOrNull(value: string): number | null {
  return value.trim() === "" ? null : Number(value);
}

export function buildHistory(identity: Identity, weeks: WeekDraft[]) {
  const midterm = weeks.find((week) => week.week === 7)?.midtermScore ?? "";
  return weeks.map((week) => {
    const assignment = numberOrNull(week.assignmentScore);
    const quiz = numberOrNull(week.quizScore);
    return {
      attempt_id: identity.attemptId.trim(),
      course_id: identity.courseId.trim().toUpperCase(),
      semester: Number(identity.semester),
      week: week.week,
      weekly_grade: Number(week.weeklyGrade),
      attendance: Number(week.attendance),
      assignment_score: assignment,
      quiz_score: quiz,
      submission_delay_days: Number(week.submissionDelayDays),
      corrections_count: Number(week.correctionsCount),
      assignment_completed: assignment === null ? 0 : 1,
      quiz_completed: quiz === null ? 0 : 1,
      midterm_score: week.week >= 7 ? numberOrNull(midterm) : null,
    };
  });
}

async function apiRequest<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.error?.message ?? payload?.detail ?? `Request failed with status ${response.status}.`;
    const fields = Array.isArray(payload?.error?.fields)
      ? payload.error.fields.map((item: { path?: string; message?: string }) => `${item.path}: ${item.message}`).join("; ")
      : "";
    throw new Error(fields || message);
  }
  return payload as T;
}

export function checkHealth(): Promise<Health> {
  return apiRequest<Health>("/api/health");
}

export function getModelInfo(): Promise<ModelInfo[]> {
  return apiRequest<ModelInfo[]>("/api/model-info");
}

export function requestPrediction(identity: Identity, weeks: WeekDraft[]): Promise<Prediction> {
  return apiRequest<Prediction>("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ history: buildHistory(identity, weeks) }),
  });
}

export type ModelUpload = {
  task: ModelInfo["task"];
  artifact: File;
  modelType: string;
  validationMetric: string;
  validationValue: string;
  rank: string;
};

export function uploadModel(upload: ModelUpload): Promise<ModelInfo> {
  const form = new FormData();
  form.set("artifact", upload.artifact);
  form.set("model_type", upload.modelType);
  form.set("validation_metric", upload.validationMetric);
  form.set("validation_value", upload.validationValue);
  form.set("rank", upload.rank);
  return apiRequest<ModelInfo>(`/api/models/${upload.task}`, { method: "POST", body: form });
}
