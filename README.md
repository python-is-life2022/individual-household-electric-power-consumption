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
- Analyze the statistical distribution of numerical features.
- Visualize trends and relationships in the data.
- Prepare the dataset for time series forecasting.
- Engineer meaningful temporal features.
- Develop and evaluate deep learning models using TensorFlow/Keras.
- Compare different forecasting approaches and model architectures.

---

## 🛠️ Technologies & Libraries

### Currently Used

- Python
- Jupyter Notebook
- Pandas — data loading, manipulation, and analysis.
- NumPy — numerical computations.
- Matplotlib — data visualization.

### To Be Added

- Seaborn — statistical visualization.
- TensorFlow/Keras — deep learning and time series forecasting.

---

## 📂 Dataset

**Name:** Individual Household Electric Power Consumption

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains measurements of household electric power consumption recorded at one-minute intervals over approximately four years.

The dataset includes measurements such as:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

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

The dataset was loaded and inspected to understand its structure, dimensions, data types, descriptive statistics, value distributions, and missing-value distribution.

---

### Stage 3 — Exploratory Data Analysis (EDA)

- [x] Analyze numerical feature distributions.
- [x] Investigate outliers using the IQR method.
- [x] Calculate quartiles and IQR-based lower and upper bounds.
- [x] Identify potential outlier observations.
- [x] Analyze skewness of `Global_active_power`.
- [x] Analyze the frequency of unique `Global_active_power` values.
- [x] Analyze the distribution of `Global_reactive_power`.
- [x] Analyze the skewness of `Global_reactive_power`.
- [x] Analyze the distribution of `Voltage`.
- [x] Investigate potential `Voltage` outliers using the IQR method.
- [x] Analyze the skewness of `Voltage`.
- [x] Analyze the distribution of `Global_intensity`.
- [x] Analyze the skewness of `Global_intensity`.
- [x] Analyze the distributions of the three sub-metering features.
- [x] Analyze the skewness of `Sub_metering_1`.
- [x] Analyze the skewness of `Sub_metering_2`.
- [x] Analyze the skewness of `Sub_metering_3`.
- [x] Calculate the correlation matrix of numerical features.
- [x] Analyze correlations with `Global_active_power`.
- [ ] Visualize electricity consumption over time.
- [ ] Identify trends and daily or seasonal patterns.
- [ ] Investigate additional relationships between numerical features.

**Notes:**

The numerical features were analyzed using descriptive statistics, value-frequency analysis, skewness analysis, and the IQR method.

For `Global_active_power`, a total of **94,907 potential statistical outliers** were identified using the 1.5 × IQR rule. The feature has a skewness of approximately **1.79**, indicating a positively skewed distribution.

For `Global_reactive_power`, the skewness was approximately **1.26**, also indicating a positively skewed distribution.

For `Voltage`, a total of **51,067 potential statistical outliers** were identified using the 1.5 × IQR rule. These observations were retained because being an IQR-based statistical outlier does not necessarily mean that a measurement is invalid. The skewness of `Voltage` was approximately **-0.33**, indicating a relatively symmetric distribution.

For `Global_intensity`, the skewness was approximately **1.85**, indicating a positively skewed distribution.

The three sub-metering features were also positively skewed:

- `Sub_metering_1`: skewness ≈ **5.94**
- `Sub_metering_2`: skewness ≈ **7.09**
- `Sub_metering_3`: skewness ≈ **0.72**

`Sub_metering_1` and `Sub_metering_2` showed particularly strong positive skewness, with a large proportion of zero-valued observations.

### Correlation Analysis

A correlation matrix was calculated to examine the linear relationships between the numerical features.

`Global_active_power` showed the following correlations with the other numerical features:

| Feature | Correlation with `Global_active_power` |
|---|---:|
| `Global_intensity` | **1.00** |
| `Sub_metering_3` | **0.64** |
| `Sub_metering_1` | **0.48** |
| `Sub_metering_2` | **0.44** |
| `Global_reactive_power` | **0.25** |
| `Voltage` | **-0.40** |

The strongest positive linear relationship was observed between `Global_active_power` and `Global_intensity`, with a correlation of approximately **1.00**.

`Sub_metering_3` showed a moderate positive correlation with `Global_active_power`, while `Sub_metering_1` and `Sub_metering_2` showed weaker positive correlations.

`Global_reactive_power` showed a relatively weak positive correlation with `Global_active_power`, whereas `Voltage` showed a moderate negative correlation.

These relationships will be considered during feature engineering and the development of the forecasting model.

#### Correlation Matrix Visualization

<!-- Add correlation matrix image here -->

![Correlation Matrix](images/correlation_matrix.png)

---

### Stage 4 — Data Cleaning & Preprocessing

- [x] Handle missing values in `Global_active_power`.
- [x] Select an appropriate imputation method based on data distribution.
- [x] Use median imputation for the skewed `Global_active_power` feature.
- [x] Verify that missing values were successfully handled.
- [x] Analyze `Global_reactive_power` skewness before imputation.
- [x] Use median imputation for `Global_reactive_power`.
- [x] Verify that missing values in `Global_reactive_power` were successfully handled.
- [x] Analyze `Voltage` skewness before imputation.
- [x] Use mean imputation for `Voltage`.
- [x] Verify that missing values in `Voltage` were successfully handled.
- [x] Analyze `Global_intensity` skewness before imputation.
- [x] Use median imputation for `Global_intensity`.
- [x] Verify that missing values in `Global_intensity` were successfully handled.
- [x] Analyze skewness of the three sub-metering features.
- [x] Use median imputation for `Sub_metering_1`.
- [x] Use median imputation for `Sub_metering_2`.
- [x] Use median imputation for `Sub_metering_3`.
- [x] Verify that all missing values were successfully handled.
- [ ] Convert date and time columns into a datetime index.
- [ ] Check data quality and consistency.
- [ ] Prepare the data for time series modeling.

**Notes:**

Missing values were handled based on the statistical distribution of each feature.

`Global_active_power`, `Global_reactive_power`, `Global_intensity`, and all three sub-metering features were positively skewed, so median imputation was selected for these features.

`Voltage` had a skewness of approximately **-0.33**, indicating a relatively symmetric distribution. Therefore, mean imputation was selected.

After the imputation process, all numerical features contain **0% missing values**.

| Feature | Imputation Method | Missing Values After Imputation |
|---|---|---:|
| `Global_active_power` | Median | 0% |
| `Global_reactive_power` | Median | 0% |
| `Voltage` | Mean | 0% |
| `Global_intensity` | Median | 0% |
| `Sub_metering_1` | Median | 0% |
| `Sub_metering_2` | Median | 0% |
| `Sub_metering_3` | Median | 0% |

The missing-value handling stage has now been completed.

---

### Stage 5 — Feature Engineering

- [ ] Convert date and time information into a datetime index.
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
- [ ] Use the Functional API for model development.
- [ ] Select an appropriate architecture for time series forecasting.
- [ ] Experiment with different sequence lengths.
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
- [ ] Analyze forecasting errors.
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
| 2026-10-28 | Stage 3 | Applied the IQR method to `Global_active_power` | 94,907 potential outliers identified |
| 2026-10-28 | Stage 3 | Calculated skewness of `Global_active_power` | Skewness ≈ 1.79 |
| 2026-10-28 | Stage 4 | Selected median imputation for `Global_active_power` | Completed |
| 2026-10-28 | Stage 4 | Imputed missing `Global_active_power` values | 0% missing values |
| 2026-10-28 | Stage 3 | Analyzed `Global_reactive_power` distribution and skewness | Skewness ≈ 1.26 |
| 2026-10-28 | Stage 4 | Selected median imputation for `Global_reactive_power` | Completed |
| 2026-10-28 | Stage 4 | Imputed missing `Global_reactive_power` values | 0% missing values |
| 2026-10-28 | Stage 3 | Applied the IQR method to `Voltage` | 51,067 potential outliers identified |
| 2026-10-28 | Stage 3 | Calculated skewness of `Voltage` | Skewness ≈ -0.33 |
| 2026-10-28 | Stage 4 | Selected mean imputation for `Voltage` | Completed |
| 2026-10-28 | Stage 4 | Imputed missing `Voltage` values | 0% missing values |
| 2026-10-28 | Stage 3 | Analyzed `Global_intensity` distribution and skewness | Skewness ≈ 1.85 |
| 2026-10-28 | Stage 4 | Selected median imputation for `Global_intensity` | Completed |
| 2026-10-28 | Stage 4 | Imputed missing `Global_intensity` values | 0% missing values |
| 2026-10-28 | Stage 3 | Analyzed the three sub-metering features | All three features were positively skewed |
| 2026-10-28 | Stage 3 | Calculated sub-metering skewness | 5.94, 7.09, and 0.72 |
| 2026-10-28 | Stage 4 | Selected median imputation for all sub-metering features | Completed |
| 2026-10-28 | Stage 4 | Verified missing values after imputation | All numerical features: 0% missing |
| 2026-10-28 | Stage 3 | Calculated correlation matrix of numerical features | Completed |
| 2026-10-28 | Stage 3 | Analyzed correlations with `Global_active_power` | Strongest correlation: `Global_intensity` ≈ 1.00 |

Add a new row whenever you complete an important experiment or make a meaningful change.

---

## 📁 Project Structure

_Update this structure as new files and folders are added._

```text
individual-household-electric-power-consumption/
├── README.md
├── images/
│   └── correlation_matrix.png
├── notebooks/
│   └── electricity_consumption_analysis.ipynb
├── data/
│   └── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

- Complete the remaining EDA steps.
- Visualize electricity consumption over time.
- Identify daily, weekly, and seasonal consumption patterns.
- Convert date and time information into a proper datetime index.
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

The initial data inspection, descriptive analysis, value-frequency analysis, outlier detection, skewness analysis, correlation analysis, and missing-value imputation have been completed.

All numerical features currently contain **0% missing values** after applying appropriate imputation strategies based on their distributions.

The next steps focus on completing the remaining exploratory data analysis, visualizing electricity consumption over time, converting the dataset into a proper time series format, engineering temporal features, splitting the data chronologically, and preparing the data for deep learning forecasting.
