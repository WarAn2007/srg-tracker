export type Stage = "intro" | "identity" | "history" | "review" | "predicting" | "results";

export type Identity = {
  attemptId: string;
  courseId: string;
  semester: string;
};

export type WeekDraft = {
  week: number;
  weeklyGrade: string;
  attendance: string;
  assignmentScore: string;
  quizScore: string;
  submissionDelayDays: string;
  correctionsCount: string;
  midtermScore: string;
};

export type Prediction = {
  student_id: string;
  course_id: string;
  semester: number;
  cutoff_week: number;
  predicted_final_gpa: number;
  predicted_course_outcome: string;
  enroll_probability: number;
  predicted_learning_pace: string;
  advisory: string;
};

export type ModelInfo = {
  task: "gpa" | "outcome" | "pace";
  model_type: string;
  validation_metric: string;
  validation_value: number;
  rank: number;
  artifact: string;
  artifact_available: boolean;
  selection_method: string;
};

export type Health = {
  status: string;
  models_ready: boolean;
  version: string;
  registry_error: string | null;
  models: ModelInfo[];
};

export type AppSettings = {
  colors: [string, string, string];
  backgroundColor: string;
  movingBackground: boolean;
};

export type HistoryEntry = {
  id: string;
  createdAt: string;
  studentId: string;
  courseId: string;
  semester: number;
  cutoffWeek: number;
  predictedGpa: number;
  outcome: string;
  pace: string;
};
