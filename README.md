# Predicting Building Damage Severity from Earthquakes Using Structural and Geographic Features
### A KDD-Based Predictive Analytics Approach

**Course:** Predictive Analytics — CA3 (End-to-End Predictive Model Development)
**Student:** Maheen Rajesh Meshram · PRN 23070521085 · B.Tech CSE, Semester VII, Section B

---

## 1. Problem Statement

On 25 April 2015 the Mw 7.8 Gorkha earthquake struck Nepal and damaged or destroyed hundreds of thousands of buildings. After an earthquake, authorities must decide quickly where to send rescue and inspection teams; before the next one, they must decide which buildings to retrofit first. Inspecting every building by hand is slow and expensive.

**Objective:** build a multi-class classification model that predicts the **damage grade** of a building — **1** (low), **2** (medium) or **3** (near-complete destruction) — using only information known *before* an earthquake (structure, materials, foundation, roof, age, size, use and location), and interpret the model to find which construction practices and regions are most vulnerable.

**Primary metric:** micro-averaged F1 (the official DrivenData metric; equals accuracy). **Secondary:** macro-F1 and per-class recall, because grade 1 is a small minority class.

## 2. Dataset

| | |
|---|---|
| Name | Richter's Predictor: Modeling Earthquake Damage (DrivenData) |
| Original source | 2015 Nepal Earthquake Open Data Portal — Central Bureau of Statistics & Kathmandu Living Labs, National Planning Commission Secretariat, Nepal |
| Link | https://www.drivendata.org/competitions/57/nepal-earthquake/ |
| Files used | `data/train_values.csv` (38 features), `data/train_labels.csv` (damage grade), joined on `building_id` |
| Size | 260,601 buildings, no missing values |
| Target | `damage_grade` — 1: 9.64 %, 2: 56.89 %, 3: 33.47 % |

Feature groups: 3 hierarchical region IDs (31 / 1,414 / 11,595 levels), 5 numeric structural features, 8 categorical construction features (obfuscated letter codes), 11 superstructure-material flags and 11 secondary-use flags. The full dictionary is in the notebook (§3.1).

## 3. Methodology — the KDD process

| # | Stage | What was done |
|---|---|---|
| 1 | Problem identification | Three-class damage-grade prediction; micro-F1 target |
| 2 | Data selection | Join values and labels one-to-one on `building_id` |
| 3 | Data understanding | Feature dictionary, cardinality, distributions, class imbalance (ratio 5.9) |
| 4 | Preprocessing | Duplicate check; 3,825 feature-identical groups with different labels (noise ceiling); `age = 995` sentinel flagged and capped; area/height winsorised at the 99.5th percentile; binary-flag consistency checks |
| 5 | EDA & hypothesis testing | Damage vs material, age, floors, foundation, roof, region; chi-square + Cramér's V, Kruskal–Wallis |
| 6 | Feature engineering | `volume_proxy`, `area_per_floor`, `height_per_floor`, `age_band`, `n_materials`, `weak_material`, `has_rc`, `families_per_floor` |
| 7 | Stratified split | 80 / 20 (208,480 / 52,121), **before** any target-dependent step |
| 8 | Transformation | Multiclass **target encoding** of region IDs with internal 5-fold cross-fitting; one-hot for categoricals; standard scaling → 83 features |
| 9 | Data mining | Dummy baseline, Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM |
| 10 | Imbalance handling | Class weights vs SMOTE (training data only) |
| 11 | Tuning | Two-stage: RandomizedSearchCV on a 60k sample → top configurations re-scored with 3-fold CV on the full training set |
| 12 | Evaluation | Micro/macro/weighted F1, per-class recall, classification report, confusion matrix, train-vs-test gap |
| 13 | Interpretation | LightGBM gain importance, SHAP summaries for grade 1 and grade 3 |
| 14 | Knowledge discovery | Calibrated destruction risk by construction profile and by region; recommendations |

## 4. Results

**Model comparison (held-out test set, 52,121 buildings)**

| Model | Micro-F1 | Macro-F1 | Recall G1 | Recall G2 | Recall G3 |
|---|---|---|---|---|---|
| Dummy (majority) | 0.5689 | 0.2417 | 0.0000 | 1.0000 | 0.0000 |
| Logistic Regression | 0.7392 | 0.6763 | 0.4697 | 0.8493 | 0.6298 |
| Decision Tree | 0.7297 | 0.6722 | 0.5095 | 0.8342 | 0.6156 |
| Random Forest | 0.7527 | 0.6998 | 0.5194 | 0.8575 | 0.6418 |
| XGBoost | 0.7513 | 0.7027 | 0.5489 | 0.8503 | 0.6415 |
| LightGBM | 0.7513 | 0.7024 | 0.5449 | 0.8497 | 0.6434 |
| LightGBM – tuned (final) | 0.7514 | 0.7021 | 0.5445 | 0.8503 | 0.6430 |

**Final model:** LightGBM (stage-1 rank 2) — **micro-F1 0.7514**, macro-F1 0.7021; full-training-set CV micro-F1 0.7438 ± 0.0013. Only 0.33 % of test buildings are confused between grade 1 and grade 3.

**Class-imbalance remedies (LightGBM)**

| Variant | Micro-F1 | Macro-F1 | Recall G1 |
|---|---|---|---|
| LightGBM – no rebalancing | 0.7513 | 0.7024 | 0.5449 |
| LightGBM – class weights | 0.7110 | 0.6860 | 0.8513 |
| LightGBM – SMOTE | 0.7048 | 0.6791 | 0.7787 |

**Destruction risk by construction profile (test set)**

| Profile | Buildings | Predicted P(grade 3) | Actual grade-3 rate |
|---|---|---|---|
| Mud-mortar stone only | 29,354 | 40.6% | 40.5% |
| Adobe / mud | 2,889 | 38.8% | 38.8% |
| Mud-mortar stone + timber | 9,349 | 32.3% | 31.9% |
| Other | 4,367 | 21.4% | 21.9% |
| RC non-engineered | 2,210 | 12.8% | 13.3% |
| Cement-mortar brick | 3,141 | 5.9% | 5.8% |
| RC engineered | 811 | 2.0% | 1.6% |

### Key findings
1. **Location dominates** — the 9 target-encoded region columns carry 77.1 % of the model's split gain; predicted and actual regional destruction rates correlate at 0.998.
2. **Mud-mortar stone is the most vulnerable common building type** (~41 % destroyed), while engineered RC buildings are rarely destroyed (~2 %).
3. **The ensembles hit a ceiling at ≈0.75 micro-F1** — identical buildings received different grades because shaking intensity and workmanship are not in the data.
4. **Class weighting is the better operating point for triage** — it lifts grade-1 recall from 54% to 85% at the cost of overall micro-F1.

### Why the model is not "100% accurate"
3,825 groups of buildings are **identical in all 38 recorded features yet received different damage grades** — neighbouring houses of the same type, age and size in the same village. What separates them is the intensity of shaking at each site, the soil and the quality of workmanship, none of which is in the data. No model can classify every building in such a group correctly, so the ceiling is set by the information, not the algorithm. This is why Random Forest, XGBoost and LightGBM all stop at about **0.75 micro-F1**, within 0.15 points of each other, and why almost all errors are between neighbouring grades.

The notebook also guards against the opposite problem — accuracy that is too good to be true. The 11,595 village IDs are target-encoded, which uses the label; encoding them naively on the full data would let each building see its own damage grade and inflate the score. Here the encoder is fitted after the train–test split, on training data only, with internal 5-fold cross-fitting, so the reported test score is honest.

## 5. Repository structure

```
├── Earthquake_Damage_Prediction_KDD.ipynb   # complete KDD notebook (executed, with outputs)
├── README.md
├── requirements.txt
├── data/
│   ├── train_values.csv
│   └── train_labels.csv
├── figures/                                 # 13 figures generated by the notebook
├── models/
│   └── final_lightgbm_pipeline.joblib       # fitted preprocessing + tuned LightGBM
├── report/
│   ├── Project_Report.pdf                   # final project report
│   ├── Project_Report.docx                  # editable version
│   └── Earthquake_Damage_Presentation.pptx  # presentation (add after download)
├── metrics.json                             # final test-set metrics
├── results_model_comparison.csv
├── results_imbalance.csv
└── results_tuning_stage2.csv
```

## 6. Report and presentation
* `report/Project_Report.pdf` — full report: Introduction, Literature Review, Workflow / Predictive Modelling Process, Results and Discussion, Conclusion, References.
* `report/Earthquake_Damage_Presentation.pptx` — presentation slides.

## 7. How to run

```bash
git clone <this-repo-url>
cd <repo>
pip install -r requirements.txt
jupyter notebook Earthquake_Damage_Prediction_KDD.ipynb   # Run All
```

A full run takes about 45 minutes on a single CPU core (most of it in tuning). All random seeds are fixed at 42, so results reproduce exactly.

**Scoring new buildings with the saved model**

```python
import joblib, pandas as pd
model = joblib.load('models/final_lightgbm_pipeline.joblib')
# X_new must contain the same columns as the notebook's FEATURES list
# (apply the same cleaning and engineer() function from §4 and §6 first)
model.predict(X_new)        # damage grade 1/2/3
model.predict_proba(X_new)  # probability of each grade
```

## 8. Limitations
* Shaking intensity, soil and workmanship are not recorded; region IDs act as a proxy for shaking, so the regional part of the model does not transfer directly to a future earthquake with a different epicentre.
* Categorical values are obfuscated by the publisher, so specific foundation/roof types cannot be named.
* The model is a screening and prioritisation tool, not a substitute for structural assessment.

## 9. References
1. Fayyad, Piatetsky-Shapiro & Smyth (1996). From data mining to knowledge discovery in databases. *AI Magazine* 17(3).
2. National Planning Commission (2015). *Nepal Earthquake 2015: Post Disaster Needs Assessment*.
3. DrivenData. Richter's Predictor: Modeling Earthquake Damage.
4. Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
5. Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS*.
6. Micci-Barreca (2001). A preprocessing scheme for high-cardinality categorical attributes. *SIGKDD Explorations* 3(1).

The full reference list is in the project report.
