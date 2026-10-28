# Individual Household Electric Power Consumption

## 📌 Project Overview

This project focuses on analyzing household electricity consumption data and developing a deep learning model to forecast future energy usage.

The project follows a step-by-step workflow, starting with data exploration and progressing toward data preprocessing, feature engineering, neural network development, and model evaluation.

**Project Status:** 🚧 In Progress

---

## 🎯 Objectives

- Analyze household electricity consumption data.
- Perform Exploratory Data Analysis (EDA).
- Identify missing values, outliers, and consumption patterns.
- Visualize trends and relationships in the data.
- Prepare time series data for forecasting.
- Develop and evaluate deep learning models using TensorFlow/Keras.

---

## 🛠️ Technologies & Libraries

### Currently Used

- Python
- Jupyter Notebook
- Pandas — data loading, manipulation, and analysis.
- Matplotlib — data visualization.

### To Be Added

- NumPy — numerical computations.
- Seaborn — statistical visualization.
- TensorFlow/Keras — deep learning and forecasting.

---

## 📂 Dataset

**Name:** Individual Household Electric Power Consumption

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains measurements of household electric power consumption recorded at one-minute intervals over approximately four years.

**Target Variable:** `Global_active_power` (kilowatts)

---

## 📈 Development Progress

This section is updated as each stage of the project is completed.

### Stage 1 — Project Setup

- [x] Create GitHub repository.
- [x] Set up Git version control.
- [x] Create the Jupyter Notebook.
- [x] Import Pandas and Matplotlib.

**Notes:**

Initial project setup and preparation of the data analysis environment.

---

### Stage 2 — Data Loading & Inspection

- [x] Load the dataset.
- [x] Inspect the dataset's dimensions and columns.
- [x] Review data types and descriptive statistics.
- [x] Identify missing values.
- [x] Calculate the percentage of missing values for each column.
- [x] Analyze value frequencies using `value_counts()`.

**Notes:**

The dataset was loaded and inspected to understand its structure, dimensions, data types, descriptive statistics, and missing-value distribution.

---

### Stage 3 — Exploratory Data Analysis (EDA)

- [x] Analyze numerical feature distributions.
- [x] Investigate outliers using the IQR method.
- [x] Calculate quartiles and IQR-based lower and upper bounds.
- [x] Identify potential outlier observations.
- [x] Analyze skewness of `Global_active_power`.
- [x] Analyze the frequency of unique `Global_active_power` values.
- [ ] Visualize electricity consumption over time.
- [ ] Identify trends and daily or seasonal patterns.
- [ ] Investigate relationships between numerical features.

**Notes:**

The `Global_active_power` feature was analyzed using descriptive statistics, value-frequency analysis, and the IQR method.

A total of **94,907 potential outliers** were identified using the 1.5 × IQR rule.

The skewness of `Global_active_power` was approximately **1.79**, indicating a positively skewed distribution. Because of the skewed distribution and the presence of potential extreme values, the median was selected as the preferred method for missing-value imputation.

---

### Stage 4 — Data Cleaning & Preprocessing

- [x] Handle missing values in `Global_active_power`.
- [x] Select an appropriate imputation method based on data distribution.
- [x] Use median imputation for the skewed `Global_active_power` feature.
- [x] Verify that missing values were successfully handled.
- [ ] Convert date and time columns into a datetime index.
- [ ] Check data quality and consistency.
- [ ] Prepare the data for time series modeling.

**Notes:**

Missing values in `Global_active_power` were replaced using the median because the feature has a positively skewed distribution.

After imputation, the missing-value percentage for `Global_active_power` was reduced to **0%**.

---

### Stage 5 — Feature Engineering

- [ ] Create time-based features.
- [ ] Extract useful information from date and time.
- [ ] Define input sequences and prediction targets.
- [ ] Split data chronologically into training, validation, and test sets.
- [ ] Normalize numerical features.
- [ ] Prepare sequences for deep learning models.

**Notes:**

_To be completed._

---

### Stage 6 — Deep Learning Model

- [ ] Build a neural network using TensorFlow/Keras.
- [ ] Select an appropriate architecture for time series forecasting.
- [ ] Train the model.
- [ ] Monitor training and validation performance.
- [ ] Tune model architecture and hyperparameters.

**Notes:**

_To be completed._

---

### Stage 7 — Model Evaluation & Forecasting

- [ ] Generate predictions on unseen data.
- [ ] Evaluate performance using suitable regression metrics.
- [ ] Visualize actual versus predicted consumption.
- [ ] Analyze model limitations.
- [ ] Compare different forecasting approaches.
- [ ] Identify possible improvements.

**Notes:**

_To be completed._

---

## 📝 Experiment Log

Use this section to record important experiments, observations, and results.

| Date | Stage | Experiment / Change | Result |
|---|---|---|---|
| 2026-10-28 | Stage 2 | Inspected dataset structure and calculated descriptive statistics | Completed |
| 2026-10-28 | Stage 2 | Analyzed `Global_active_power` value frequencies | Completed |
| 2026-10-28 | Stage 3 | Applied the IQR method to detect potential outliers | 94,907 potential outliers identified |
| 2026-10-28 | Stage 3 | Calculated skewness of `Global_active_power` | Skewness ≈ 1.79 |
| 2026-10-28 | Stage 4 | Selected median based on the skewed distribution | Completed |
| 2026-10-28 | Stage 4 | Imputed missing `Global_active_power` values using the median | Completed |
| 2026-10-28 | Stage 4 | Verified missing values after imputation | `Global_active_power`: 0% missing |

Add a new row whenever you complete an important experiment or make a meaningful change.

---

## 📁 Project Structure

_Update this structure as new files and folders are added._

```text
individual-household-electric-power-consumption/
├── README.md
├── notebooks/
│   └── electricity_consumption_analysis.ipynb
├── data/
│   └── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

- Complete the remaining EDA steps.
- Create time-based features for time series forecasting.
- Experiment with different sequence lengths.
- Compare different neural network architectures.
- Experiment with LSTM, GRU, and other deep learning approaches.
- Compare forecasting performance across models.
- Improve feature engineering and preprocessing.
- Explore more advanced time series forecasting techniques.
- Analyze model limitations and forecasting errors.

---

## 📌 Project Status

This project is actively being developed.

The initial data inspection, descriptive analysis, outlier detection, skewness analysis, and missing-value imputation have been completed.

The next steps focus on completing the exploratory data analysis, preparing the dataset as a time series, engineering temporal features, and developing a deep learning forecasting model.
