# Cervical Cancer Risk Prediction: Traditional ML vs Deep Learning

A comparative study that predicts **biopsy-confirmed cervical cancer risk** from
non-invasive patient risk factors, benchmarking classical Scikit-learn models against
TensorFlow neural networks (Sequential API, Functional API, and `tf.data` pipelines).

> **Motivation.** Cervical cancer is largely preventable, yet most deaths occur in
> low- and middle-income countries where screening tests and specialists are scarce.
> This project asks a focused question: *can a model triage risk from a questionnaire
> alone, flagging high-risk women before invasive testing?*

---

## Links

- 📄 **Report (IEEE):** _[ add link to PDF in this repo or a drive link ]_
- 🎥 **Demo video (5 min):** _[ add link ]_
- 📓 **Notebook:** [`notebooks/Cervical_Cancer_Risk_Prediction.ipynb`](notebooks/Cervical_Cancer_Risk_Prediction.ipynb)
- 📊 **Dataset:** UCI Cervical Cancer (Risk Factors) — Fernandes, Cardoso & Fernandes (2017), DOI [10.24432/C5Z310](https://doi.org/10.24432/C5Z310)

---

## Problem & Data

| | |
|---|---|
| **Task** | Binary classification — predict biopsy outcome |
| **Records** | 858 patients |
| **Features** | 32 non-invasive risk factors (demographics, reproductive history, smoking, contraceptive use, STD history, prior diagnoses) |
| **Target** | `Biopsy` (histological gold standard) |
| **Class balance** | **6.4% positive** (55 / 858) — severe imbalance |

The dataset is deliberately messy: structured missingness (two columns ~92% empty) and
heavy class imbalance, which makes accuracy a misleading metric and drives the focus on
**recall, ROC-AUC, and PR-AUC**.

---

## Approach

**Preprocessing & feature engineering**
- Dropped two ~92%-empty columns and the three non-target screening outcomes (leakage control)
- Train-only median/zero imputation and standardization (no test leakage)
- Stratified 80/20 split preserving the 6.4% positive rate
- 7 domain-engineered features (STD burden, years sexually active, contraceptive exposure, etc.)

**Eight systematic experiments**

| # | Model | Key change |
|---|-------|-----------|
| 1 | Logistic Regression | Baseline (exposes the accuracy trap) |
| 2 | Logistic Regression | Balanced class weights |
| 3 | Random Forest | Nonlinear baseline |
| 4 | Random Forest | Tuned via GridSearchCV (best classical) |
| 5 | NN — Sequential API | Baseline + `tf.data` |
| 6 | NN — Sequential API | Class weights + dropout + L2 |
| 7 | NN — Functional API | Regularized, batch-normalized variant |
| 8 | NN — Functional API | Decision-threshold tuning for screening |

Neural networks are trained over **5 random seeds** (mean ± std) for stability, since only
37 training positives make single runs unreliable.

---

## Results (held-out test set: 11 positives, 161 negatives)

| Experiment | Acc | Recall | ROC-AUC | PR-AUC | Caught |
|---|---|---|---|---|---|
| LogReg (baseline) | 0.936 | 0.000 | 0.578 | 0.094 | 0/11 |
| LogReg (balanced) | 0.762 | 0.273 | 0.562 | 0.076 | 3/11 |
| Random Forest (default) | 0.942 | 0.091 | 0.652 | **0.186** | 1/11 |
| RF (balanced + tuned) | 0.919 | 0.091 | 0.649 | 0.130 | 1/11 |
| DL Sequential (weighted+reg) | 0.855 | 0.182 | 0.61* | 0.09* | 2/11 |
| DL Functional (weighted+reg) | 0.866 | 0.182 | 0.65* | 0.11* | 2/11 |
| DL threshold-tuned | 0.785 | 0.364 | 0.61* | 0.09* | 4/11 |

\* deep-learning values are 5-seed means; reproduce with the notebook.

**Key findings**
1. **Classical ML matches or beats deep learning** on this small, imbalanced tabular data (RF PR-AUC 0.186 vs best DL ~0.11).
2. **Accuracy is misleading** — both paradigms hit ~94% accuracy with ~0 recall by default.
3. **Threshold tuning** is essential to turn any model into a usable screening tool.
4. **Risk factors alone are weak predictors** of biopsy outcome — a clinically meaningful result.
5. **Engineered features carry real signal** — three rank in the Random Forest's top five.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Cervical_Cancer_Risk_Prediction.ipynb
├── figures/
│   ├── fig1_learning_curves.png
│   ├── fig2_confusion.png
│   ├── fig3_roc_pr.png
│   └── fig4_importance.png
└── report/
    └── Cervical_Cancer_Report.pdf
```

---

## Reproduce

The notebook runs top-to-bottom in Google Colab (T4 GPU recommended) with no manual
downloads — the dataset is streamed from a public mirror.

```bash
pip install -r requirements.txt
# then open the notebook and Run All
```

Random seeds are fixed for the classical models and the data split; deep-learning results
are reported as a mean across five seeds.

---

## Citation

> K. Fernandes, J. Cardoso, and J. Fernandes, "Cervical Cancer (Risk Factors),"
> UCI Machine Learning Repository, 2017. doi: 10.24432/C5Z310.

## Author

**[ Your Name ]** — BSc Software Engineering (ML), African Leadership University.

## License

Released under the MIT License (see `LICENSE`). Dataset used under CC BY 4.0.
