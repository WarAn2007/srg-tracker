import { useEffect, useMemo, useState } from "react";

import { checkHealth, requestPrediction } from "./api";
import CreditsDialog from "./components/CreditsDialog";
import HistoryDialog from "./components/HistoryDialog";
import Lightfall from "./components/Lightfall";
import SettingsDialog from "./components/SettingsDialog";
import { loadSettings, saveSettings } from "./settings";
import type { AppSettings, Health, Identity, Prediction, Stage, WeekDraft } from "./types";

const EMPTY_IDENTITY: Identity = { attemptId: "", courseId: "", semester: "" };

function emptyWeek(week: number): WeekDraft {
  return { week, weeklyGrade: "", attendance: "", assignmentScore: "", quizScore: "", submissionDelayDays: "0", correctionsCount: "0", midtermScore: "" };
}

function useSystemReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    update(); media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return reduced;
}

function numericError(label: string, value: string, min: number, max: number, optional = false) {
  if (!value.trim()) return optional ? "" : `${label} is required.`;
  const number = Number(value);
  return Number.isFinite(number) && number >= min && number <= max ? "" : `${label} must be between ${min} and ${max}.`;
}

function validateWeek(week: WeekDraft) {
  return numericError("Weekly grade", week.weeklyGrade, 0, 100)
    || numericError("Attendance", week.attendance, 0, 100)
    || numericError("Assignment score", week.assignmentScore, 0, 100, true)
    || numericError("Quiz score", week.quizScore, 0, 100, true)
    || numericError("Submission delay", week.submissionDelayDays, 0, 30)
    || numericError("Corrections", week.correctionsCount, 0, 100)
    || (week.week === 7 ? numericError("Midterm score", week.midtermScore, 0, 100) : "");
}

function NumberField({ label, value, onChange, min = 0, max = 100, step = "0.1", optional = false, hint }: { label: string; value: string; onChange: (value: string) => void; min?: number; max?: number; step?: string; optional?: boolean; hint?: string }) {
  return <label className="field"><span>{label}{optional && <em>optional</em>}</span><input type="number" inputMode="decimal" min={min} max={max} step={step} value={value} onChange={(event) => onChange(event.target.value)} placeholder={`${min}–${max}`} />{hint && <small>{hint}</small>}</label>;
}

function StepRail({ stage }: { stage: Stage }) {
  const active = stage === "identity" ? 0 : stage === "history" ? 1 : stage === "review" || stage === "predicting" ? 2 : stage === "results" ? 3 : -1;
  return <ol className="step-rail" aria-label="Prediction workflow">{["Identity", "Weekly history", "Review", "Results"].map((label, index) => <li key={label} className={index === active ? "active" : index < active ? "complete" : ""}><span>{index < active ? "✓" : index + 1}</span><b>{label}</b></li>)}</ol>;
}

export default function App() {
  const systemReducedMotion = useSystemReducedMotion();
  const [settings, setSettings] = useState<AppSettings>(loadSettings);
  const [stage, setStage] = useState<Stage>("intro");
  const [identity, setIdentity] = useState<Identity>(EMPTY_IDENTITY);
  const [weeks, setWeeks] = useState<WeekDraft[]>([emptyWeek(1)]);
  const [activeWeek, setActiveWeek] = useState(1);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [dialog, setDialog] = useState<"settings" | "credits" | "history" | null>(null);
  const [toast, setToast] = useState("");

  const currentWeek = weeks.find((week) => week.week === activeWeek) ?? weeks[0];
  const pausedBackground = systemReducedMotion || !settings.movingBackground;

  useEffect(() => { saveSettings(settings); }, [settings]);
  const refreshHealth = () => checkHealth()
    .then(setHealth)
    .catch(() => setHealth({ status: "offline", models_ready: false, registry_error: "FastAPI is not reachable.", version: "4.0", models: [] }));

  useEffect(() => { void refreshHealth(); }, []);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(""), 5200); return () => window.clearTimeout(timer); }, [toast]);

  const preAssessment = useMemo(() => ({
    assignment: weeks.some((week) => week.week < 7 && week.assignmentScore !== ""),
    quiz: weeks.some((week) => week.week < 7 && week.quizScore !== ""),
  }), [weeks]);

  const updateWeek = (field: keyof WeekDraft, value: string) => setWeeks((current) => current.map((week) => week.week === activeWeek ? { ...week, [field]: value } : week));

  const registerIdentity = (event: React.FormEvent) => {
    event.preventDefault();
    if (!identity.attemptId.trim()) return setToast("Student ID is required.");
    if (!identity.courseId.trim()) return setToast("Course ID is required.");
    const semester = Number(identity.semester);
    if (!Number.isInteger(semester) || semester < 1 || semester > 20) return setToast("Semester must be a whole number from 1 to 20.");
    setStage("history");
  };

  const addWeek = () => {
    if (activeWeek !== weeks.length) return setToast(`Open week ${weeks.length} before adding the next week.`);
    const error = validateWeek(currentWeek);
    if (error) return setToast(`Week ${activeWeek}: ${error}`);
    if (activeWeek === 6 && (!preAssessment.assignment || !preAssessment.quiz)) return setToast("Record at least one assignment and one quiz before week 7.");
    if (weeks.length >= 14) return setToast("Week 14 is the final observation week.");
    const next = weeks.length + 1;
    setWeeks((current) => [...current, emptyWeek(next)]); setActiveWeek(next);
  };

  const openReview = () => {
    for (const week of weeks) {
      const error = validateWeek(week);
      if (error) { setActiveWeek(week.week); setToast(`Week ${week.week}: ${error}`); return; }
    }
    if (weeks.length >= 7 && (!preAssessment.assignment || !preAssessment.quiz)) return setToast("Record at least one assignment and one quiz before week 7.");
    setStage("review");
  };

  const predict = async () => {
    if (!health?.models_ready) return setToast(health?.registry_error ?? "Packaged models are not ready.");
    setStage("predicting");
    try {
      const result = await requestPrediction(identity, weeks);
      setPrediction(result); setStage("results");
    } catch (error) {
      setStage("review"); setToast(error instanceof Error ? error.message : "Prediction failed.");
    }
  };

  const reset = () => {
    setIdentity(EMPTY_IDENTITY); setWeeks([emptyWeek(1)]); setActiveWeek(1); setPrediction(null); setStage("identity");
  };

  const handleSettings = (next: AppSettings) => setSettings(next);
  const healthLabel = health?.models_ready ? "3 models ready" : health ? "Model attention required" : "Checking models";

  return (
    <main className={`app-shell stage-${stage}`} style={{ "--accent": settings.colors[1], "--highlight": settings.colors[2] } as React.CSSProperties}>
      <Lightfall colors={settings.colors} backgroundColor={settings.backgroundColor} paused={pausedBackground} />
      <div className="field-shade" />

      <header className="topbar">
        <button className="brand" onClick={() => setStage("intro")} aria-label="Go to SRG-Tracker home"><span className="brand-glyph">S</span><span><b>SRG-Tracker</b><small>Student signal observatory</small></span><em>4.0</em></button>
        <nav aria-label="Application utilities"><button onClick={() => setDialog("history")}>History</button><button onClick={() => setDialog("credits")}>Credits</button><button onClick={() => setDialog("settings")}>Settings</button></nav>
        <div className={`health-chip ${health?.models_ready ? "ready" : "warning"}`}><i />{healthLabel}</div>
      </header>

      {stage !== "intro" && <StepRail stage={stage} />}

      {stage === "intro" && <section className="hero">
        <div className="hero-copy"><p className="eyebrow">Observed history → advisory signals</p><h1>See the course<br />before it <span>ends.</span></h1><p className="hero-lead">Enter consecutive weekly observations and receive a careful estimate of final GPA, course outcome, and learning pace.</p><div className="hero-actions"><button className="primary-button" onClick={() => setStage("identity")}>Start assessment <span>→</span></button><button className="quiet-button" onClick={() => setDialog("credits")}>How this works</button></div><p className="responsibility-line"><i />Educational demonstration · Synthetic-data models · Human review required</p></div>
        <aside className="signal-card"><div className="signal-card-header"><span>Active model registry</span><b>{health?.models_ready ? "READY" : "CHECKING"}</b></div>{health?.models?.map((model) => <article key={model.task}><span>{model.task === "gpa" ? "Final GPA" : model.task === "outcome" ? "Course outcome" : "Learning pace"}</span><strong>{model.model_type}</strong><small>{model.validation_metric.replace("_", " ")} {model.validation_value.toFixed(3)} · rank {model.rank}</small></article>)}{health && !health.models_ready && <p className="registry-error">{health.registry_error}</p>}<button onClick={() => setDialog("settings")}>View model information</button></aside>
      </section>}

      {stage === "identity" && <section className="workspace"><div className="workspace-heading"><p className="eyebrow">Step 1 of 4</p><h1>Identify the course attempt</h1><p>These labels appear in the final report and encrypted history. They are not model features except course and semester.</p></div><form className="identity-form" onSubmit={registerIdentity}><label className="field wide"><span>Student ID</span><input autoFocus value={identity.attemptId} onChange={(event) => setIdentity({ ...identity, attemptId: event.target.value })} placeholder="S0001-PY101-1" /><small>Use a demo identifier, never real personal data.</small></label><label className="field"><span>Course ID</span><input value={identity.courseId} onChange={(event) => setIdentity({ ...identity, courseId: event.target.value })} placeholder="PY101" /></label><label className="field"><span>Semester</span><input type="number" min="1" max="20" step="1" value={identity.semester} onChange={(event) => setIdentity({ ...identity, semester: event.target.value })} placeholder="1" /></label><div className="form-footer"><button type="button" className="text-button" onClick={() => setStage("intro")}>Back</button><button className="primary-button" type="submit">Continue to weekly history →</button></div></form></section>}

      {stage === "history" && <section className="workspace history-workspace"><div className="workspace-heading compact"><div><p className="eyebrow">Step 2 of 4 · {identity.courseId.toUpperCase()} · semester {identity.semester}</p><h1>Build the observed history</h1></div><div className="week-summary"><strong>{weeks.length}</strong><span>week{weeks.length === 1 ? "" : "s"}<br />observed</span></div></div><nav className="week-tabs" aria-label="Observed weeks">{weeks.map((week) => <button key={week.week} className={week.week === activeWeek ? "active" : ""} aria-current={week.week === activeWeek ? "step" : undefined} onClick={() => setActiveWeek(week.week)}>W{String(week.week).padStart(2, "0")}</button>)}</nav><div className="week-heading"><div><span>Week {activeWeek}</span><h2>{activeWeek < 7 ? "Pre-midterm observation" : activeWeek === 7 ? "Midterm checkpoint" : "Post-midterm observation"}</h2></div><p>Only information observed by this week is allowed.</p></div><div className="data-grid"><NumberField label="Weekly grade" value={currentWeek.weeklyGrade} onChange={(value) => updateWeek("weeklyGrade", value)} /><NumberField label="Attendance" value={currentWeek.attendance} onChange={(value) => updateWeek("attendance", value)} /><NumberField label="Assignment score" optional value={currentWeek.assignmentScore} onChange={(value) => updateWeek("assignmentScore", value)} /><NumberField label="Quiz score" optional value={currentWeek.quizScore} onChange={(value) => updateWeek("quizScore", value)} /><NumberField label="Submission delay" value={currentWeek.submissionDelayDays} onChange={(value) => updateWeek("submissionDelayDays", value)} min={0} max={30} hint="Days" /><NumberField label="Corrections" value={currentWeek.correctionsCount} onChange={(value) => updateWeek("correctionsCount", value)} step="1" />{activeWeek === 7 && <NumberField label="Midterm score" value={currentWeek.midtermScore} onChange={(value) => updateWeek("midtermScore", value)} hint="Reused unchanged after week 7" />}{activeWeek > 7 && <div className="locked-value"><span>Midterm score</span><strong>{weeks.find((week) => week.week === 7)?.midtermScore || "—"}</strong><small>Locked from week 7</small></div>}</div><div className="form-footer"><button className="secondary-button" onClick={addWeek} disabled={weeks.length >= 14 || activeWeek !== weeks.length}>+ Add next week</button><button className="primary-button" onClick={openReview}>Review input →</button></div></section>}

      {stage === "review" && <section className="workspace review-workspace"><div className="workspace-heading compact"><div><p className="eyebrow">Step 3 of 4</p><h1>Review before prediction</h1></div><span className="validation-badge">✓ Input structure valid</span></div><div className="identity-summary"><div><span>Student</span><strong>{identity.attemptId}</strong></div><div><span>Course</span><strong>{identity.courseId.toUpperCase()}</strong></div><div><span>Semester</span><strong>{identity.semester}</strong></div><div><span>Cutoff</span><strong>Week {weeks.length}</strong></div></div><div className="review-table-wrap"><table><thead><tr><th>Week</th><th>Grade</th><th>Attendance</th><th>Assignment</th><th>Quiz</th><th>Delay</th></tr></thead><tbody>{weeks.map((week) => <tr key={week.week}><td>{week.week}</td><td>{week.weeklyGrade}</td><td>{week.attendance}%</td><td>{week.assignmentScore || "—"}</td><td>{week.quizScore || "—"}</td><td>{week.submissionDelayDays}d</td></tr>)}</tbody></table></div>{weeks.length < 7 && <div className="evidence-note"><strong>Early evidence context</strong><p>This prediction uses fewer than seven weeks and no midterm. Treat its confidence as limited and update it when more observations are available.</p></div>}<div className="form-footer"><button className="text-button" onClick={() => setStage("history")}>Edit weekly history</button><button className="primary-button" onClick={predict}>Run advisory prediction →</button></div></section>}

      {stage === "predicting" && <section className="processing" aria-live="polite"><div className="processing-mark"><i /><i /><i /></div><p className="eyebrow">Models are reading the observed signal</p><h1>Preparing an advisory view…</h1><p>No training is running. The packaged models are already loaded in memory.</p></section>}

      {stage === "results" && prediction && <section className="workspace results-workspace"><div className="result-intro"><div><p className="eyebrow">Step 4 of 4 · Friendly result</p><h1>{prediction.student_id}, here is the current course signal.</h1><p>For <strong>{prediction.course_id}</strong>, semester <strong>{prediction.semester}</strong>, using observations through week <strong>{prediction.cutoff_week}</strong>.</p></div><span className="result-seal">ADVISORY<br />ONLY</span></div><div className="result-grid"><article><span>Estimated final GPA</span><strong>{prediction.predicted_final_gpa.toFixed(2)}</strong><small>Scale 0.00–4.50</small></article><article><span>Estimated course outcome</span><strong className={prediction.predicted_course_outcome}>{prediction.predicted_course_outcome}</strong><small>{Math.round(prediction.enroll_probability * 100)}% model probability for enroll · not causal certainty</small></article><article><span>Estimated learning pace</span><strong>{prediction.predicted_learning_pace.replace("_", " ")}</strong><small>Pattern-based label from observed history</small></article></div><div className="advisory-panel"><span>Human review note</span><p>{prediction.advisory}</p></div><div className="form-footer result-actions"><button className="secondary-button" onClick={() => setDialog("history")}>Save to encrypted history</button><button className="primary-button" onClick={reset}>Assess another student →</button></div></section>}

      {dialog === "settings" && <SettingsDialog settings={settings} onSettings={handleSettings} onClose={() => setDialog(null)} onNotice={setToast} onModelActivated={() => void refreshHealth()} />}
      {dialog === "credits" && <CreditsDialog onClose={() => setDialog(null)} />}
      {dialog === "history" && <HistoryDialog prediction={prediction} onClose={() => setDialog(null)} onNotice={setToast} />}
      {toast && <div className="toast" role="alert"><span>{toast}</span><button onClick={() => setToast("")} aria-label="Dismiss notification">×</button></div>}
      <footer className="app-footer">Student ranking is out of scope. <span>Predictions require qualified human review.</span></footer>
    </main>
  );
}
