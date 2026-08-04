# Future update ideas

The following ideas are intentionally deferred so V4.0 can remain a focused, reliable local release.

## 1. Weekly signal chart

Add an accessible chart for weekly grade, attendance, assignment, and quiz trends. The chart should use only observed weeks, include a table alternative, and never suggest that correlation is causation.

## 2. Local report export

Export one advisory result as a printable PDF and structured JSON. The report should include model metadata, cutoff week, uncertainty language, and the responsible-use statement without embedding the encrypted history password.

## 3. Change since the previous prediction

Compare a student's current prediction with their own previous encrypted record. Show changes in GPA estimate, outcome probability, and pace without comparing students or creating a rank.

## 4. History management

Add deletion of individual encrypted records, password rotation, optional merge conflict handling during import, and integrity/version migration for future `.srg-history` formats.

## 5. Stronger model provenance

Add a signed deployment bundle containing the three model files, manifest, checksums, training code version, dataset version, and validation report. Require signature verification before activation.

## 6. Calibration and evidence communication

Add task-specific calibration summaries and evidence-quality labels for different cutoff weeks. This requires new evaluation evidence for the exact deployed model mapping and must not reuse historical metrics after a model change.
