import { useRef, useState } from "react";

import {
  decryptHistory,
  downloadEnvelope,
  encryptHistory,
  hasLocalVault,
  importLocalEnvelope,
  parseEnvelopeFile,
  readLocalEnvelope,
  saveLocalHistory,
  type VaultEnvelope,
} from "../historyVault";
import type { HistoryEntry, Prediction } from "../types";
import Modal from "./Modal";

function entryFromPrediction(prediction: Prediction): HistoryEntry {
  return {
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    studentId: prediction.student_id,
    courseId: prediction.course_id,
    semester: prediction.semester,
    cutoffWeek: prediction.cutoff_week,
    predictedGpa: prediction.predicted_final_gpa,
    outcome: prediction.predicted_course_outcome,
    pace: prediction.predicted_learning_pace,
  };
}

export default function HistoryDialog({ prediction, onClose, onNotice }: { prediction: Prediction | null; onClose: () => void; onNotice: (message: string) => void }) {
  const [unlocked, setUnlocked] = useState(false);
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const importRef = useRef<HTMLInputElement>(null);
  const exists = hasLocalVault();

  const unlockOrCreate = async () => {
    setBusy(true); setError("");
    try {
      if (exists) {
        const envelope = readLocalEnvelope();
        if (!envelope) throw new Error("The local history vault could not be read.");
        setEntries(await decryptHistory(envelope, password));
      } else {
        if (password !== confirmation) throw new Error("The two passwords do not match.");
        await saveLocalHistory([], password);
        setEntries([]);
      }
      setUnlocked(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "History vault could not be opened.");
    } finally { setBusy(false); }
  };

  const savePrediction = async () => {
    if (!prediction) return;
    const entry = entryFromPrediction(prediction);
    const next = [entry, ...entries];
    await saveLocalHistory(next, password);
    setEntries(next);
    onNotice("Prediction saved to encrypted history.");
  };

  const exportFile = async () => {
    downloadEnvelope(await encryptHistory(entries, password));
    onNotice("Encrypted history file exported.");
  };

  const importFile = async (file: File) => {
    setError("");
    try {
      const envelope = parseEnvelopeFile(await file.arrayBuffer()) as VaultEnvelope;
      const importedEntries = await decryptHistory(envelope, password);
      importLocalEnvelope(envelope);
      setEntries(importedEntries);
      setUnlocked(true);
      onNotice("Encrypted history file imported.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "History import failed.");
    }
  };

  return (
    <Modal eyebrow="Encrypted local records" title="Prediction history" onClose={onClose} wide>
      {!unlocked ? (
        <div className="vault-gate">
          <div className="vault-symbol" aria-hidden="true">◇</div>
          <h3>{exists ? "Unlock your local vault" : "Create your local vault"}</h3>
          <p>Records are encrypted with AES-GCM before they are stored. The password never leaves this browser and cannot be recovered.</p>
          <label><span>Password</span><input type="password" autoComplete="off" value={password} onChange={(event) => setPassword(event.target.value)} /></label>
          {!exists && <label><span>Confirm password</span><input type="password" autoComplete="off" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} /></label>}
          {error && <p className="inline-error" role="alert">{error}</p>}
          <button className="primary-button" disabled={busy} onClick={unlockOrCreate}>{busy ? "Opening…" : exists ? "Unlock history" : "Create encrypted history"}</button>
          <button className="text-button" onClick={() => importRef.current?.click()}>Import an encrypted `.srg-history` file</button>
          <input ref={importRef} className="visually-hidden" type="file" accept=".srg-history,application/octet-stream" onChange={(event) => event.target.files?.[0] && importFile(event.target.files[0])} />
        </div>
      ) : (
        <div className="history-view">
          <div className="history-toolbar">
            <div><strong>{entries.length}</strong><span>encrypted record{entries.length === 1 ? "" : "s"}</span></div>
            <div>
              {prediction && <button className="secondary-button" onClick={savePrediction}>Save current result</button>}
              <button className="secondary-button" onClick={exportFile}>Export file</button>
              <button className="secondary-button" onClick={() => importRef.current?.click()}>Import file</button>
              <input ref={importRef} className="visually-hidden" type="file" accept=".srg-history,application/octet-stream" onChange={(event) => event.target.files?.[0] && importFile(event.target.files[0])} />
            </div>
          </div>
          {error && <p className="inline-error" role="alert">{error}</p>}
          {entries.length === 0 ? <div className="empty-history"><h3>No saved predictions yet</h3><p>Run a prediction, then use “Save current result.”</p></div> : (
            <div className="history-table-wrap"><table className="history-table"><thead><tr><th>Date</th><th>Student</th><th>Course</th><th>Semester</th><th>Week</th><th>GPA</th><th>Outcome</th><th>Pace</th></tr></thead><tbody>{entries.map((entry) => <tr key={entry.id}><td>{new Date(entry.createdAt).toLocaleDateString()}</td><td>{entry.studentId}</td><td>{entry.courseId}</td><td>{entry.semester}</td><td>{entry.cutoffWeek}</td><td>{entry.predictedGpa.toFixed(2)}</td><td>{entry.outcome}</td><td>{entry.pace.replace("_", " ")}</td></tr>)}</tbody></table></div>
          )}
        </div>
      )}
    </Modal>
  );
}
