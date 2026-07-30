# SRG-Tracker — Project Start Log

**Date:** 26 July 2026  
**Project:** Student Stats / Early Warning Model  
**Author:** Anvar Ibragimov  
**Repository:** https://github.com/WarAn2007/srg-tracker

## Purpose of the discussion

The first project session was used to understand the Capstone requirements, define the ML problem, agree on the dataset design, and prepare a reproducible Git repository.

## Initial project idea

The project is an educational early-warning system for students. It should use information available during a course to estimate whether a student is progressing adequately and whether the student may need to repeat the course.

The first implementation will be a Python/Jupyter project. A later notebook or demo should accept a student-course record and return an estimated final course score, course outcome, and learning pace.

## Requirements reviewed

The following documents were reviewed:

- `docx/Capstone_Project_Implementation.docx`
- `docx/Capstone_Project_Evaluation_Criteria.docx`
- `docx/Capstone_Project.docx`

Important Capstone requirements identified:

- a trained or fine-tuned ML model is required;
- evaluation must use unseen data;
- a baseline and at least two meaningful approaches must be compared;
- the repository must contain preprocessing, evaluation, error analysis, README and a reproducible demo;
- the project must document limitations, privacy, fairness and responsible use;
- a clean Google Colab/runtime workflow is expected.

## Decisions agreed with the student

### Targets

The project will use three supervised learning tasks:

1. **Final course score regression** — predict `final_course_score` on a 0–100 scale.
2. **Course outcome classification** — predict `pass` or `enroll`. Here, `enroll` means that the student did not pass and must repeat the course.
3. **Learning pace classification** — predict `behind`, `on_track`, or `ahead`.

### Final-score formula

The final score for one course is defined as:

```text
0.25 × midterm_score
+ 0.25 × final_exam_score
+ 0.10 × attendance_score
+ 0.10 × average_weekly_grade
+ 0.20 × average_assignment_score
+ 0.20 × average_quiz_score
```

The course outcome threshold is:

```text
pass     if final_course_score >= 65
enroll   otherwise
```

The final-exam score is not an input before the final exam. It is used only to construct or audit the final target, preventing target leakage.

### Weekly data design

Each student must have a record for every week. One row represents:

```text
student × course × semester × week
```

The generated dataset uses:

- 450 unique students;
- 2 semesters per student;
- 5 courses per semester;
- 14 weekly snapshots per course;
- 63,000 rows in total.

The main weekly features include attendance, current average grades, assignment and quiz averages, submission delays, correction count, completed activities, semester, course and week. The midterm score appears only after it becomes available.

### Data split

The split is performed by unique student, not by random rows, to prevent the same student appearing in multiple splits:

- train: 300 students / 42,000 rows;
- validation: 100 students / 14,000 rows;
- test: 50 students / 7,000 rows.

The test split must remain untouched until final evaluation.

### Evaluation metrics

- Final course score: MAE, RMSE and R², because the target is continuous.
- Course outcome: recall for identifying students at risk of repeating, F1 for the precision/recall balance, and ROC-AUC when probabilities are available.
- Learning pace: macro F1 and balanced accuracy because the three classes may be imbalanced.
- Cross-validation is a training/validation procedure, not a metric.

### Product scope

In scope:

- weekly predictions for score, outcome and learning pace;
- synthetic-data generation;
- EDA, preprocessing and leakage prevention;
- baseline and model comparison;
- error analysis and reproducible notebook demo;
- input validation and advisory warnings.

Out of scope for the first version:

- real LMS data collection;
- automatic grading or enrollment decisions;
- student ranking or disciplinary decisions;
- friend matching;
- personalised causal recommendations;
- production deployment, authentication and persistent user accounts.

## Actions completed on 26 July

- Initialized the local Git repository and renamed the branch to `main`.
- Created the initial project structure, README, `.gitignore`, requirements file and data metadata.
- Created a deterministic Python dataset generator with seed `20260726`.
- Generated train, validation and test CSV files with the agreed student-disjoint split.
- Checked row counts, student overlap, score ranges and target categories.
- Updated `docx/Capstone_Project.docx`, including the technical proposal, metrics, scope, acceptance criteria, unresolved questions and optional directions.
- Created the first two local commits.
- Created and connected the GitHub repository `WarAn2007/srg-tracker`.
- Pushed the `main` branch to GitHub.

## Limitations recorded

The dataset is synthetic and generated from simplified domain rules. It does not represent real student behaviour and must not be used for actual academic decisions. Any future use with real LMS data would require privacy review, legal permission, representativeness checks and human oversight.

## Next planned actions

1. Build EDA and preprocessing notebooks.
2. Train baselines and multiple supervised models.
3. Compare validation results and select final candidates.
4. Evaluate once on the held-out test split.
5. Perform error analysis by week, course and outcome.
6. Build a clean Colab-compatible inference/demo notebook.
7. Document experiment results and final limitations in the README.
