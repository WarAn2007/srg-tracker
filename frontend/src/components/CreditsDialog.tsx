import Modal from "./Modal";

export default function CreditsDialog({ onClose }: { onClose: () => void }) {
  return (
    <Modal eyebrow="About this project" title="Credits & interface guide" onClose={onClose} wide>
      <div className="credits-layout">
        <article className="author-card">
          <span className="author-monogram">IA</span>
          <p className="overline">Created by</p>
          <h3>Ibragimov Anvar</h3>
          <p>Beginner in AI/ML and Robotics Engineering.</p>
        </article>
        <div className="credits-copy">
          <section>
            <h3>Purpose</h3>
            <p>SRG-Tracker turns one student’s observed, consecutive weekly course history into three advisory estimates: final GPA, course outcome, and learning pace. It supports an informed conversation with an instructor; it never ranks students or makes automatic academic decisions.</p>
          </section>
          <section>
            <h3>What each control does</h3>
            <dl className="control-guide">
              <div><dt>Start assessment</dt><dd>Begins a new student and course attempt.</dd></div>
              <div><dt>Add week</dt><dd>Validates the current week before opening the next consecutive week.</dd></div>
              <div><dt>Review</dt><dd>Summarizes identity and weekly observations before prediction.</dd></div>
              <div><dt>Settings</dt><dd>Changes the Lightfall palette, motion preference, and trusted model artifacts.</dd></div>
              <div><dt>History</dt><dd>Unlocks, imports, exports, and displays encrypted local prediction records.</dd></div>
              <div><dt>Model information</dt><dd>Shows the active artifact, validation metric, rank, and selection method.</dd></div>
            </dl>
          </section>
        </div>
      </div>
    </Modal>
  );
}
