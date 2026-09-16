# Instructor Effectiveness Modeling 

Defining, scoring, and predicting instructor effectiveness from batch-level Tech data — with an emphasis on justifying every modeling choice and being upfront about where the approach could mislead or fail in the real world.

## Problem

The dataset has no ground-truth "effectiveness" label — only raw batch-level metrics (completion, dropout, quiz scores, engagement, feedback, etc.). The task is to:
1. Define a defensible effectiveness score from the available signals
2. Aggregate batch-level data to the instructor level
3. Train a model to predict an effectiveness tier
4. Interpret the model and be honest about its limitations

## Dataset

`instructor_effectiveness_dataset.csv` — 2,000 batch-level rows covering 120 instructors and 25 courses. Each row is one course batch; the same instructor can appear across many batches.

| Column | Description |
|---|---|
| `batch_id`, `instructor_id`, `course_id` | Identifiers |
| `completion_rate`, `dropout_rate` | Outcome metrics |
| `avg_score_improvement`, `avg_quiz_score` | Learning outcome metrics |
| `avg_watch_time`, `assignment_submission_rate`, `forum_activity_rate` | Engagement metrics |
| `avg_feedback_score`, `feedback_response_rate` | Feedback metrics |

No missing values or duplicate batch IDs — the data was already clean at the batch level.

## Approach

1. **EDA** — distributions, correlations, and consistency checks (e.g. completion and dropout are near-complements, as expected).
2. **Instructor Effectiveness Score** — a composite, min-max normalized score built from three weighted pillars:
   - **Outcomes (50%)** — completion, dropout, score improvement, quiz score
   - **Engagement (25%)** — watch time, assignment submission, forum activity
   - **Feedback (25%)** — feedback score, feedback response rate
3. **Aggregation to instructor level** — mean of each metric per instructor, plus `n_batches` (batch count) and `score_std` (consistency) to preserve reliability information that a mean alone would hide.
4. **Effectiveness Tiers** — the continuous score is split into **Low / Medium / High** using tertile cut-offs (chosen over fixed thresholds since there's no external absolute benchmark for "good").
5. **Model** — a Random Forest Classifier trained on the 120 instructor-level rows to predict tier, reaching **~92% test accuracy / ~0.92 macro-F1**.
6. **Interpretation** — feature importance shows outcome metrics (completion, dropout, feedback response) drive the model far more than raw engagement metrics.

## Key findings

- **Completion rate, dropout rate, and feedback response rate** are the strongest predictors of tier — whether learners finish, stay, and bother to give feedback at all matters more than how actively they engage along the way.
- Engagement metrics (watch time, forum activity) contribute comparatively little.
- The score's own construction (50% outcomes weight) is echoed by what the model *discovers* to be important — a useful internal consistency check.

## Known limitations (see notebook for full discussion)

- **Label circularity** — the tier is derived from the same features used to predict it, so the model mostly reproduces the scoring formula rather than validating an independent notion of effectiveness.
- **Small sample** — 120 instructors (96 train / 24 test) means results can be sensitive to the train/test split.
- **No course/cohort controls** — instructors aren't randomly assigned to courses, so a "Low" tier could reflect a harder course rather than weaker teaching.
- **Feedback self-selection bias** — only learners who choose to respond shape `avg_feedback_score` and `feedback_response_rate`.
- **Static snapshot** — no way to detect an instructor trending up or down over time.

**Bottom line:** this is a reasonable **triage/support tool** (e.g. flagging batches or instructors worth a closer look) but should not be used on its own for high-stakes decisions like pay or termination, especially not across different courses.

## Repo structure

```
├── instructor_effectiveness_dataset.csv   # Raw batch-level dataset
├── Instructor_Effectiveness_Modeling.ipynb # Full analysis notebook
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Author

Shivam Rawat
