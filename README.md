# mimic2-mortality-prediction
# Time-Series Feature Engineering and Mortality Prediction with MIMIC-II

This project demonstrates how to process and model time-series ICU data from the PhysioNet 2012 Challenge dataset, which is derived from MIMIC-II but structured specifically for the challenge task. It covers everything from raw .txt files per patient to engineered features, imputation, and model comparison.


## 🔍 Overview

* **Language:** R
* **Libraries:** `data.table`, `tidyverse`, `tidymodels`, `skimr`, `naniar`, `bonsai`, `themIS`
* **Dataset:** [The PhysioNet/Computing in Cardiology Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/)
---

## 📂 Project Workflow

### 1. 🔢 Data Exploration

* Read a sample of patient `.txt` files to understand structure
* Count how frequently each variable appears
* Extract the full list of unique variables across all subjects

### 2. 💩 Data Cleaning & Feature Engineering

#### 2.1 Stack Raw Files

* Read all individual `.txt` files
* Convert timestamps to minutes since ICU admission
* Combine into one long-format table: `recordid`, `time_min`, `parameter`, `value`

#### 2.2 Define Variable Groups

* **Static variables** (non-time-series):  
  Age, Gender, Height, ICUType, Weight

* **Time-series variables** (measured over the first 48h after addmission):  
  * **Frequent** — measured frequently (10+ times/day):  
    HR, GCS, NISysABP, NIMAP, RespRate, Temp, Urine, DiasABP, FiO₂, MAP, SysABP, NIDiasABP
  * **Moderate** — measured a few times per day:  
    K, Mg, Na, WBC, PaCO₂, PaO₂, BUN, Creatinine, Glucose, HCO₃, pH, HCT, Weight
  * **Rare** — only ordered when clinically indicated:  
    Platelets, ALP, ALT, AST, Albumin, Cholesterol, Lactate, TroponinI, TroponinT, Bilirubin, SaO₂
  * **Categorical** — time-varying binary indicator:  
    MechVent (mechanical ventilation status)


#### 2.3 Feature Extraction

Time-series variables were engineered based on their clinical measurement frequency and information content:

* **Static variables**: Extracted baseline values at time = 0.

---

**Continuous Time-Series Variables**

* **Frequent variables** (10+/day):  
  - **Features**: Median, IQR, maximum, minimum, trend, number of observations (`num_obs`), first value, and missing indicators  
  - **Aggregation**: Computed across three time windows — 0–6h, 6–24h, and 24–48h — to retain temporal granularity and reflect physiological evolution.

* **Moderate variables** (1–3/day):  
  - **Features**: Same as frequent variables  
  - **Aggregation**: Over the entire 48-hour period due to lower sampling frequency.

* **Rare variables** (only ordered as needed):  
  - **Features**: Median, trend, `num_obs`, first value, and missing indicator  
  - **Aggregation**: Over the entire 48-hour period.

---

**Categorical Time-Series Variable**

* **MechVent** (mechanical ventilation status):  
  - This is the only time-varying categorical variable. Since only `MechVent = 1` was recorded (indicating ventilation was on), the absence of values implies no ventilation.
  - **Derived features**:
    - `n_obs`: total number of recorded ventilation events
    - `on`: binary indicator (1 if ventilated at any point, 0 if never)
    - `first`, `last`: timestamps of the first and last MechVent observations
    - `duration_cat`: ventilation duration (last − first) categorized into 5 bins:
      - 0 = not ventilated  
      - 1 = short (≤ 10 hours)  
      - 2 = moderate (10–30 hours)  
      - 3 = long (30–42 hours)  
      - 4 = extended (> 42 hours)

---

**Additional Notes:**
- *First value*: Often reflects the patient’s baseline status before therapeutic interventions, which can be highly predictive.
- *Trend*: Defined as `last - first` value; captures the direction and magnitude of change.
- *Missing indicators*: Informative in themselves — absence of data may signal lack of monitoring or clinical concern.
  

#### 2.4 Finalize Dataset

* Pivot all features into wide format
* Merge with outcome labels

---

## 🧼 Missing Data Handling

* All `_missing` indicators were retained as features
* For **rare variables**, missing values were assumed to reflect normal test results (clinical assumption)
* Imputed using the **midpoint of normal clinical ranges**, stratified by gender when necessary
* Patients with >300 missing predictors were excluded

---

## 🧰 Predictive Modeling

### 1. Preprocessing

* Converted categorical variables to factors
* Removed irrelevant outcome variables (e.g., SOFA, SAPS-I)
* Split data into training (70%) and test (30%) using stratification on death
* Applied SMOTE to address class imbalance

### 2. Models Developed

#### ✅ **Elastic Net**

* Tuned `penalty` and `mixture` via grid search (5-fold CV)
* Balanced regularization yielded strong results (F1 = 0.506)

#### ✅ **Lasso**

* Special case of Elastic Net (`mixture = 1`)
* Achieved **best overall performance** across F1 and sensitivity

#### ✅ **Random Forest**

* Tuned `mtry` and `min_n` via Bayesian optimization
* High AUC (0.848), but low recall (F1 = 0.145)

#### ✅ **LightGBM**

* Tuned multiple hyperparameters (tree depth, learning rate, L1/L2 penalty) via Bayesian optimization
* Fast and accurate but relatively low sensitivity

#### ✅ **XGBoost**

* Tuned multiple hyperparameters (tree depth, learning rate, loss reduction, etc) via Bayesian optimization
* Slower learner; performed worse than LightGBM in this task

---

### 📊 Model Comparison

| Model         | AUC   | Accuracy | Sensitivity | PPV   | F1 Score  |
| ------------- | ----- | -------- | ----------- | ----- | --------- |
| Lasso         | 0.863 | 0.799    | 0.751       | 0.389 | **0.513** |
| Elastic Net   | 0.868 | 0.791    | 0.760       | 0.379 | 0.506     |
| LightGBM      | 0.875 | 0.883    | 0.297       | 0.699 | 0.417     |
| XGBoost       | 0.806 | 0.855    | 0.181       | 0.462 | 0.260     |
| Random Forest | 0.848 | 0.867    | 0.080       | 0.771 | 0.145     |

> F1 score was chosen as the primary evaluation metric due to class imbalance.

---

## 🔎 Interpretation of Best Model (Lasso)

Key predictors selected by Lasso included:

* **Age** (+): Older age increases risk
* **GCS\_max\_24** (–): Neurologic function is highly predictive
* **BUN\_min**, **Creatinine\_median**, **Bilirubin\_median** (+): Indicators of renal/liver dysfunction
* **Na\_trend**, **HCO3\_trend** (+/–): Metabolic trends were informative
* **Missingness indicators** (e.g., RespRate\_missing\_0, NISysABP\_missing\_24): Missing data can itself be a signal of poor monitoring or patient condition

---

## 📄 Files

* `data_cleaning_feature_engineering.R`: Combines data ingestion, cleaning, and feature engineering
* `modeling.R`: Contains all modeling code and hyperparameter tuning



