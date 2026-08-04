import { useEffect, useState } from "react";

import { getModelInfo, uploadModel } from "../api";
import { DEFAULT_SETTINGS } from "../settings";
import type { AppSettings, ModelInfo } from "../types";
import Modal from "./Modal";

type UploadDraft = {
  task: ModelInfo["task"];
  artifact: File | null;
  modelType: string;
  validationMetric: string;
  validationValue: string;
  rank: string;
};

const INITIAL_UPLOAD: UploadDraft = {
  task: "gpa",
  artifact: null,
  modelType: "linear",
  validationMetric: "rmse",
  validationValue: "",
  rank: "1",
};

export default function SettingsDialog({
  settings,
  onSettings,
  onClose,
  onNotice,
  onModelActivated,
}: {
  settings: AppSettings;
  onSettings: (settings: AppSettings) => void;
  onClose: () => void;
  onNotice: (message: string) => void;
  onModelActivated: () => void;
}) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [upload, setUpload] = useState<UploadDraft>(INITIAL_UPLOAD);
  const [uploading, setUploading] = useState(false);
  const [modelError, setModelError] = useState("");

  const refresh = () => getModelInfo().then(setModels).catch((error: Error) => setModelError(error.message));
  useEffect(() => { void refresh(); }, []);

  const updateColor = (index: number, value: string) => {
    const colors = [...settings.colors] as [string, string, string];
    colors[index] = value;
    onSettings({ ...settings, colors });
  };

  const submitModel = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!upload.artifact) return setModelError("Choose a trusted .joblib artifact.");
    if (!upload.validationValue.trim()) return setModelError("Enter the validation value recorded during model selection.");
    setUploading(true);
    setModelError("");
    try {
      const result = await uploadModel({ ...upload, artifact: upload.artifact });
      setModels((current) => current.map((item) => item.task === result.task ? result : item));
      setUpload({ ...INITIAL_UPLOAD, task: upload.task });
      onModelActivated();
      onNotice(`${result.task.toUpperCase()} model validated and activated.`);
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Model replacement failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <Modal eyebrow="Local preferences" title="Settings" onClose={onClose} wide>
      <div className="settings-grid">
        <section className="settings-section appearance-settings">
          <div className="section-title"><span>01</span><div><h3>Signal palette</h3><p>Four independent color rings tune the Lightfall field.</p></div></div>
          <div className="color-rings">
            {settings.colors.map((color, index) => (
              <label key={index} className="color-ring">
                <input type="color" value={color} onChange={(event) => updateColor(index, event.target.value)} />
                <span>{["Primary", "Accent", "Highlight"][index]}</span>
                <code>{color.toUpperCase()}</code>
              </label>
            ))}
            <label className="color-ring">
              <input type="color" value={settings.backgroundColor} onChange={(event) => onSettings({ ...settings, backgroundColor: event.target.value })} />
              <span>Ambient glow</span>
              <code>{settings.backgroundColor.toUpperCase()}</code>
            </label>
          </div>
          <div className="motion-row">
            <div><strong>Moving background</strong><p>Pause Lightfall while keeping the selected colors.</p></div>
            <button
              className={`switch ${settings.movingBackground ? "on" : ""}`}
              style={{ "--switch-color": settings.colors[1] } as React.CSSProperties}
              role="switch"
              aria-checked={settings.movingBackground}
              onClick={() => onSettings({ ...settings, movingBackground: !settings.movingBackground })}
            ><span /></button>
          </div>
          <button className="text-button" onClick={() => onSettings(DEFAULT_SETTINGS)}>Reset appearance to default</button>
        </section>

        <section className="settings-section model-settings">
          <div className="section-title"><span>02</span><div><h3>Active models</h3><p>Each task can use a different compatible pipeline.</p></div></div>
          <div className="model-list">
            {models.map((model) => (
              <article key={model.task} className="model-card">
                <div><span>{model.task}</span><strong>{model.model_type}</strong></div>
                <p>{model.validation_metric}: {model.validation_value.toFixed(4)} · rank {model.rank}</p>
                <small>{model.artifact}</small>
              </article>
            ))}
          </div>
          <form className="model-upload" onSubmit={submitModel}>
            <div className="trust-warning"><strong>Trusted files only.</strong> A `.joblib` file can execute Python code while loading. Use only artifacts you created or reviewed.</div>
            <div className="form-row">
              <label><span>Task</span><select value={upload.task} onChange={(event) => setUpload({ ...upload, task: event.target.value as ModelInfo["task"] })}><option value="gpa">GPA</option><option value="outcome">Outcome</option><option value="pace">Pace</option></select></label>
              <label><span>Model type</span><input value={upload.modelType} onChange={(event) => setUpload({ ...upload, modelType: event.target.value })} /></label>
            </div>
            <div className="form-row three-columns">
              <label><span>Metric</span><input value={upload.validationMetric} onChange={(event) => setUpload({ ...upload, validationMetric: event.target.value })} /></label>
              <label><span>Value</span><input type="number" step="any" value={upload.validationValue} onChange={(event) => setUpload({ ...upload, validationValue: event.target.value })} /></label>
              <label><span>Rank</span><input type="number" min="1" step="1" value={upload.rank} onChange={(event) => setUpload({ ...upload, rank: event.target.value })} /></label>
            </div>
            <label className="file-field">
              <span>Artifact</span>
              <input className="visually-hidden" type="file" accept=".joblib" onChange={(event) => setUpload({ ...upload, artifact: event.target.files?.[0] ?? null })} />
              <span className="file-select"><b>Choose .joblib</b><em>{upload.artifact?.name ?? "No file selected"}</em></span>
            </label>
            {modelError && <p className="inline-error" role="alert">{modelError}</p>}
            <button className="primary-button compact" type="submit" disabled={uploading}>{uploading ? "Validating artifact…" : "Validate and activate model"}</button>
          </form>
        </section>
      </div>
    </Modal>
  );
}
