# Patient Health Records – Missing Values & Outlier Handling

A data preprocessing project on **patient health records**. The notebook generates a realistic dataset with missing values and outliers, then cleans it using multiple imputation and outlier-handling techniques so it is ready for a machine learning model that predicts **disease risk** (`disease_risk`: 0 = Low, 1 = High).

> **Role:** Data Analyst at a healthcare company
> **Focus:** Missing value imputation (Part A) and outlier detection/treatment (Part B)

---

## Project Objective

Prepare noisy patient health data for downstream ML by:
- Measuring and filling missing values with several imputation strategies
- Detecting outliers with Z-score, IQR, Percentile and Winsorization methods
- Comparing dataset shape and summary statistics before and after treatment

---

## Folder Structure

```
Second PR Feature Engineering/
├── Secd.ipynb                          # Main notebook (data generation + Part A + Part B)
├── patient_health_records.csv          # Generated dataset (1000 rows x 9 columns)
├── Screenshot 2026-09-29 151921.png    # Output screenshots (13 total)
│   ...
├── Screenshot 2026-09-29 152126.png
└── README.md
```

---

## Dataset Description

The dataset is generated in the notebook with NumPy (`seed = 42`) and saved as `patient_health_records.csv`. It has **1000 patients** and **9 columns**.

| Column | Type | Description | Missing Values |
|---|---|---|---|
| patient_id | String | Unique patient ID (P0001 – P1000) | 0 |
| age | Integer | Patient age (18–90) | 50 (5%) |
| gender | Categorical | Male / Female | 60 (6%) |
| region | Categorical | North / South / East / West | 70 (7%) |
| bmi | Float | Body Mass Index | 80 (8%) |
| blood_pressure | Float | Blood pressure reading | 0 |
| cholesterol | Float | Cholesterol level | 100 (10%) |
| glucose | Float | Blood glucose level | 90 (9%) |
| disease_risk | Binary | **Target:** 0 = No risk, 1 = At risk | 0 |

**Target balance:** 683 patients with `disease_risk = 0` and 317 with `disease_risk = 1` (about 31.7%).

**Injected data issues:** roughly 3% extreme values were added to `bmi`, `blood_pressure`, `cholesterol` and `glucose`, and missing values were randomly introduced in six columns.

---

## Part A – Handling Missing Values

### 1. Missing Value Summary
Counted missing values with `df.isnull().sum()` and converted them to percentages per column: `age` 5%, `gender` 6%, `region` 7%, `bmi` 8%, `cholesterol` 10%, `glucose` 9%.
Screenshots: `151951`, `152005`

### 2. Simple Imputer
- **Numerical (`bmi`):** `SimpleImputer(strategy="median")`, since BMI has extreme values and the median is robust to them
- **Categorical (`region`):** `SimpleImputer(strategy="most_frequent")`

Screenshot: `152019`

### 3. KNN Imputer
- Numeric columns (`age`, `bmi`, `blood_pressure`, `cholesterol`, `glucose`) standardized with `StandardScaler` first, because KNN is distance-based
- `KNNImputer(n_neighbors=5)` applied, then values converted back to real units with `inverse_transform`
- Gender and region filled with the mode
- Result: **0 missing values**

Screenshot: `152029`

### 4. MICE Algorithm (Iterative Imputer)
- `IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)` on the numeric columns
- Age rounded back to whole numbers; gender and region filled with the mode
- Result: **0 missing values**

Screenshot: `152036`

The MICE-imputed dataset (`df_clean`, shape `(1000, 9)`) is the base used for outlier handling in Part B. (`152047`)

---

## Part B – Handling Outliers

### 1. Z-score Method (Cholesterol & Glucose)
Flagged values with `|z| > 3`.

| Column | Outliers Found |
|---|---|
| cholesterol | 23 |
| glucose | 30 |

Rows flagged in either column were removed: **1000 → 948 rows**. (`152053`)

### 2. IQR Method (BMI)
- Normal BMI range: **15.3 to 32.9** (`Q1 − 1.5×IQR` to `Q3 + 1.5×IQR`)
- BMI outliers found: **38**
- After removal: **1000 → 962 rows**

Screenshot: `152100`

### 3. Percentile Method
Capped `bmi`, `blood_pressure`, `cholesterol` and `glucose` at the **1st and 99th percentiles** using `clip()`. All 1000 rows are kept. (`152109`)

### 4. Winsorization
Applied `scipy.stats.mstats.winsorize` with limits of 1% on each side to the same four columns. All 1000 rows are kept. (`152117`)

### 5. Before vs After Comparison

| Method | Resulting Shape | What Happens to Outliers |
|---|---|---|
| Original (after MICE) | (1000, 9) | – |
| Z-score removal | (948, 9) | Rows removed |
| IQR removal | (962, 9) | Rows removed |
| Percentile capping | (1000, 9) | Values capped |
| Winsorization | (1000, 9) | Values capped |

Effect of capping on the four columns (Before → Percentile / Winsorized):

| Column | Min (Before → After) | Max (Before → After) | Std (Before → After) |
|---|---|---|---|
| bmi | 11.60 → 16.20 | 64.70 → 61.80 | 6.51 → 6.37 |
| blood_pressure | 72.70 → 88.39 | 258.50 → 243.60 | 23.06 → 22.43 |
| cholesterol | 46.20 → 104.51 | 440.40 → 405.47 | 42.93 → 41.25 |
| glucose | 50.30 → 60.49 | 395.00 → 367.61 | 43.93 → 42.95 |

Means stay almost unchanged (for example BMI 24.95 → 24.95), while the extreme minimum and maximum values are pulled in. Percentile capping and Winsorization give nearly identical results.

Screenshot: `152126`

---

## Screenshots

| Step | Preview |
|---|---|
| Dataset generation output | ![Dataset generation](Screenshot%202026-09-29%20151921.png) |
| Dataset preview and shape | ![Dataset preview](Screenshot%202026-09-29%20151942.png) |
| Missing value counts | ![Missing counts](Screenshot%202026-09-29%20151951.png) |
| Missing value percentages | ![Missing percentages](Screenshot%202026-09-29%20152005.png) |
| Simple Imputer | ![Simple Imputer](Screenshot%202026-09-29%20152019.png) |
| KNN Imputer | ![KNN Imputer](Screenshot%202026-09-29%20152029.png) |
| MICE | ![MICE](Screenshot%202026-09-29%20152036.png) |
| Cleaned base for Part B | ![Part B setup](Screenshot%202026-09-29%20152047.png) |
| Z-score method | ![Z-score](Screenshot%202026-09-29%20152053.png) |
| IQR method | ![IQR](Screenshot%202026-09-29%20152100.png) |
| Percentile method | ![Percentile](Screenshot%202026-09-29%20152109.png) |
| Winsorization | ![Winsorization](Screenshot%202026-09-29%20152117.png) |
| Before vs after comparison | ![Comparison](Screenshot%202026-09-29%20152126.png) |

---

## Tech Stack

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (`SimpleImputer`, `KNNImputer`, `IterativeImputer`, `BayesianRidge`, `StandardScaler`)
- SciPy (`winsorize`)
- Jupyter Notebook

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
jupyter notebook Secd.ipynb
```

Run all cells in order. The first code cell generates and saves `patient_health_records.csv`, which the later cells read back.

---

## Author

**Subhash**
BCA (Semester 5), Veer Narmad South Gujarat University (VNSGU)
Aspiring AI/ML practitioner
