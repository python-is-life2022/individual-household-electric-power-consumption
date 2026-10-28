# ⚡ Household Energy Forecasting

```{=html}
<p align="center">
```
`<strong>`{=html}Exploratory Data Analysis · Time-Series Forecasting ·
Deep Learning`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">`{=html}
`<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">`{=html}
`<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">`{=html}
`<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">`{=html}
`<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square" alt="Project Status">`{=html}
`<img src="https://img.shields.io/badge/Task-Time%20Series%20Forecasting-2563EB?style=flat-square" alt="Task">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

This project explores household electricity consumption data and
develops a deep learning pipeline for forecasting future energy usage.

The workflow covers data inspection, exploratory data analysis,
missing-value handling, temporal feature engineering, chronological
splitting, feature scaling, neural network development, and initial
model training.

The initial fully connected neural network has been trained for 10
epochs using validation data. Further evaluation and the development of
a proper sequence-based forecasting setup remain in progress.

## 🎯 Objectives

-   🔎 Perform exploratory data analysis (EDA).
-   🧹 Identify and handle missing values.
-   📊 Investigate feature distributions, skewness, outliers, and
    correlations.
-   🕒 Analyze temporal patterns and consumption trends.
-   ⚙️ Engineer temporal features and prepare forecasting inputs.
-   📏 Apply appropriate feature scaling without data leakage.
-   🧠 Develop deep learning models using TensorFlow/Keras.
-   📉 Monitor training and validation performance.
-   📈 Evaluate and compare forecasting performance.

## 📂 Dataset

**Individual Household Electric Power Consumption**

Source: [UCI Machine Learning
Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains approximately four years of household electricity
measurements recorded at one-minute intervals.

  Feature                   Description
  ------------------------- --------------------------------------------
  `Date`                    Measurement date
  `Time`                    Measurement time
  `Global_active_power`     Total active power consumption
  `Global_reactive_power`   Total reactive power
  `Voltage`                 Average voltage
  `Global_intensity`        Average current intensity
  `Sub_metering_1`          Energy sub-metering for kitchen appliances
  `Sub_metering_2`          Energy sub-metering for laundry appliances
  `Sub_metering_3`          Energy sub-metering for other appliances

The current model uses `Global_active_power` as its regression target.
The forecasting horizon and sequence-based input structure still need to
be established before the model can be considered a complete time-series
forecasting solution.

## 🛠️ Tech Stack

  Tool                   Purpose
  ---------------------- ---------------------------------------------
  Python                 Core programming language
  Jupyter Notebook       Interactive development
  Pandas & NumPy         Data manipulation and numerical computation
  Matplotlib & Seaborn   Visualization and statistical analysis
  Scikit-learn           Preprocessing and data splitting
  TensorFlow / Keras     Neural network development and training

------------------------------------------------------------------------

## 🔬 Project Workflow

### 01 · Data Loading & Inspection

-   Loaded the dataset into a Pandas DataFrame.
-   Examined dataset dimensions, column names, and data types.
-   Calculated descriptive statistics.
-   Inspected missing values, unique values, and value frequencies.

### 02 · Exploratory Data Analysis

Performed distribution analysis, skewness analysis, potential outlier
detection using the Interquartile Range (IQR) method, and correlation
analysis.

#### Potential Outliers

  Feature                   Number of Potential Outliers
  ----------------------- ------------------------------
  `Global_active_power`                           94,907
  `Voltage`                                       51,067

These observations were flagged by the IQR method and were not
automatically removed.

#### Feature Skewness

  Feature                     Skewness
  ------------------------- ----------
  `Global_active_power`         \~1.79
  `Global_reactive_power`       \~1.26
  `Voltage`                    \~-0.33
  `Global_intensity`            \~1.85
  `Sub_metering_1`              \~5.94
  `Sub_metering_2`              \~7.09
  `Sub_metering_3`              \~0.72

#### Correlation with Global Active Power

  Feature                     Correlation
  ------------------------- -------------
  `Global_intensity`               \~1.00
  `Sub_metering_3`                 \~0.64
  `Sub_metering_1`                 \~0.48
  `Sub_metering_2`                 \~0.44
  `Global_reactive_power`          \~0.25
  `Voltage`                       \~-0.40

The correlation matrix was saved to:

`images/correlation_matrix.png`

```{=html}
<p align="center">
```
`<img src="images/correlation_matrix.png" alt="Correlation matrix of household electricity consumption features" width="800">`{=html}
```{=html}
</p>
```
### 03 · Missing Value Handling

Missing values in numerical features were handled using statistical
imputation.

**Median imputation** was applied to skewed consumption-related
features:

-   `Global_active_power`
-   `Global_reactive_power`
-   `Global_intensity`
-   `Sub_metering_1`
-   `Sub_metering_2`
-   `Sub_metering_3`

**Mean imputation** was applied to `Voltage`, whose observed
distribution had relatively low skewness.

After imputation, all numerical features contained **0% missing
values**.

### 04 · Temporal Feature Engineering

The `Date` and `Time` columns were combined into a single `datetime`
column using:

``` python
df["datetime"] = pd.to_datetime(
    df["Date"] + " " + df["Time"],
    format="%d/%m/%Y %H:%M:%S"
)
```

The following temporal features were extracted:

-   `Year`
-   `Month`
-   `Day`
-   `DayOfWeek`
-   `Hour`
-   `Minute`

`DayOfWeek` uses values from `0` (Monday) to `6` (Sunday).

The original `datetime` column was retained for temporal analysis and
future sequence generation.

### 05 · Chronological Data Splitting

The dataset was split chronologically using Scikit-learn's
`train_test_split` with `shuffle=False`.

  Dataset        Proportion    Observations
  ------------ ------------ ---------------
  Training              72%       1,494,186
  Validation             8%         166,021
  Test                  20%         415,052
  **Total**        **100%**   **2,075,259**

**Temporal order:** Past → Training → Validation → Test → Future

Preserving temporal order helps prevent future observations from
entering the training set. Additional care is required when generating
sequences and aligning input observations with future targets.

### 06 · Feature Scaling

Numerical electricity measurements were standardized using
Scikit-learn's `StandardScaler`.

To prevent data leakage, the scaler was fitted only on the training set.

  Dataset      Operation
  ------------ -------------------
  Training     `fit_transform()`
  Validation   `transform()`
  Test         `transform()`

The standardized features include:

-   `Global_active_power`
-   `Global_reactive_power`
-   `Voltage`
-   `Global_intensity`
-   `Sub_metering_1`
-   `Sub_metering_2`
-   `Sub_metering_3`

Temporal features will be handled separately. Cyclical encoding may be
applied to periodic features such as `Hour`, `DayOfWeek`, and `Month`.

### 07 · Initial Neural Network

An initial fully connected neural network was defined using the **Keras
Functional API**.

#### Architecture

``` text
Input (6 features)
       │
       ▼
Dense (20 neurons, ReLU)
       │
       ▼
Dropout (10%)
       │
       ▼
Dense (15 neurons, ReLU)
       │
       ▼
Dropout (15%)
       │
       ▼
Dense (1 neuron, Linear)
```

#### Layer Configuration

  Layer     Configuration      Purpose
  --------- ------------------ ----------------------------------
  Input     6 features         Accept numerical inputs
  Dense     20 neurons, ReLU   Learn nonlinear patterns
  Dropout   0.10               Regularization
  Dense     15 neurons, ReLU   Learn additional representations
  Dropout   0.15               Regularization
  Output    1 neuron           Produce a continuous prediction

#### Compilation Configuration

  Parameter       Value
  --------------- ---------------------------
  Optimizer       Adam
  Learning rate   `0.0001`
  Loss function   Mean Squared Error (MSE)
  Metric          Mean Absolute Error (MAE)

MSE is used as the optimization objective, while MAE measures the
average absolute difference between predicted and actual values.

The architecture serves as an initial regression baseline. Its current
six-feature input represents the other standardized numerical
measurements, while the output represents `Global_active_power`.

### 08 · Initial Model Training

The compiled neural network was trained for **10 epochs** using a batch
size of **32** and a separate validation dataset.

#### Training Configuration

  Parameter                           Value
  ----------------------------------- --------------------------
  Epochs                              `10`
  Batch size                          `32`
  Input features                      6 numerical measurements
  Target                              `Global_active_power`
  Optimizer                           Adam
  Learning rate                       `0.0001`
  Loss function                       MSE
  Evaluation metric during training   MAE

#### Training Implementation

``` python
his = model.fit(
    x=train_norm[:, 1:],
    y=train_norm[:, 0],
    validation_data=(
        val_norm[:, 1:],
        val_norm[:, 0]
    ),
    epochs=10,
    batch_size=32
)
```

The first column of the standardized dataset, `Global_active_power`, is
used as the target. The remaining six columns are supplied as input
features.

Validation data is passed separately through `validation_data`, allowing
training and validation losses and MAE to be monitored across epochs.

The training history is stored in `his`, which can be used to visualize
the learning curves:

``` python
his.history.keys()
```

The history can subsequently be used to plot training and validation
loss and MAE to investigate convergence and potential overfitting.

> **Important:** This initial training run establishes a regression
> baseline. Since the current inputs are measurements from the same
> observation rather than explicitly generated historical sequences,
> this step alone does not establish future-consumption forecasting
> performance.

------------------------------------------------------------------------

### 09 · Model Evaluation

The initial neural network was evaluated on both the training and test
datasets using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

#### Evaluation Results

  Metric         Training Set   Test Set
  ------------ -------------- ----------
  Loss (MSE)           0.0088     0.0067
  MAE                  0.0790     0.0681

#### Results Analysis

The model achieved lower loss and MAE values on the test dataset than on
the training dataset. This is an encouraging observation, but it does
not necessarily mean the model generalizes better. The difference may be
influenced by the dataset split, preprocessing, or characteristics of
the samples.

These results are an initial regression baseline, not yet a definitive
measure of future forecasting performance. The model inputs currently
use measurements from the same observation rather than explicitly
generated historical sequences.

#### Evaluation Implementation

``` python
train_loss, train_mae = model.evaluate(
    train_norm[:, 1:],
    train_norm[:, 0]
)

test_loss, test_mae = model.evaluate(
    test_norm[:, 1:],
    test_norm[:, 0]
)

print(f"Train Loss: {train_loss:.4f}")
print(f"Train MAE: {train_mae:.4f}")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
```

Because the target was standardized, the MAE values shown here are in
the standardized target scale, not directly in kilowatts (kW).
Evaluation on the original target scale and a sequence-based forecasting
setup remain important next steps.

------------------------------------------------------------------------

## 📓 Experiment Log

### ✅ Completed

-   [x] Dataset loading and initial inspection
-   [x] Missing-value analysis
-   [x] Descriptive statistics and feature distributions
-   [x] Skewness and potential outlier analysis
-   [x] Correlation analysis and visualization
-   [x] Statistical missing-value imputation
-   [x] Datetime creation and temporal feature extraction
-   [x] Chronological training, validation, and test split
-   [x] Numerical feature scaling using training data only
-   [x] Initial Functional API model definition
-   [x] Model compilation with Adam, MSE, and MAE
-   [x] Initial model training for 10 epochs
-   [x] Validation data supplied during training
-   [x] Evaluate model performance on the training dataset
-   [x] Evaluate model performance on the test dataset
-   [x] Compare training and test loss and MAE

### 🚧 In Progress

-   [ ] Analyze training and validation loss curves
-   [ ] Compare training and validation MAE
-   [ ] Assess model convergence and potential overfitting
-   [ ] Define the future forecasting target and prediction horizon
-   [ ] Align model inputs and targets with the forecasting objective
-   [ ] Analyze predictions and evaluate errors on the original target
    scale

### 🔜 Planned

-   [ ] Visualize daily, weekly, and seasonal consumption patterns
-   [ ] Apply cyclical encoding where appropriate
-   [ ] Generate historical input-output sequences
-   [ ] Experiment with forecasting-specific neural network
    architectures
-   [ ] Tune model hyperparameters
-   [ ] Evaluate performance on the held-out test set
-   [ ] Calculate MAE, MSE, and RMSE
-   [ ] Visualize actual versus predicted consumption over time
-   [ ] Compare forecasting approaches and document the results

## 📊 Evaluation Strategy

The initial model has been evaluated on the training and test datasets
using MSE and MAE. The current results provide an initial comparison of
regression performance, but further analysis is needed before drawing
conclusions about future electricity consumption forecasting.

The next evaluation steps include:

-   Analyze training and validation learning curves.
-   Visualize actual versus predicted values.
-   Calculate evaluation metrics on the original target scale.
-   Define historical input sequences and a forecasting horizon.
-   Evaluate forecasting performance on unseen future observations.
-   Compare different neural network architectures and hyperparameters.

The final forecasting model will be assessed using MAE, MSE, and RMSE,
alongside visual comparisons of actual and predicted electricity
consumption.

## 📁 Repository Structure

``` text
.
├── images/
│   └── correlation_matrix.png
├── notebooks/
│   └── electricity_consumption.ipynb
├── README.md
└── requirements.txt
```

*The repository structure above is illustrative; update the notebook
path to match the actual project files.*

## 📌 Project Status

**Current stage:** Initial Model Evaluation and Forecasting Development

The project has progressed from data exploration and preprocessing to
training and evaluating an initial neural network regression model.
Initial training and test metrics have been recorded.

The next milestone is to analyze the model's predictions, evaluate
errors on the original consumption scale, and establish a sequence-based
forecasting pipeline.

## 📚 References

-   [UCI Machine Learning Repository --- Individual Household Electric
    Power
    Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
-   [TensorFlow Documentation](https://www.tensorflow.org/)
-   [Keras Documentation](https://keras.io/)
-   [Scikit-learn Documentation](https://scikit-learn.org/)

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<sub>`{=html}Developed as a hands-on machine learning and deep learning
project.`</sub>`{=html}
```{=html}
</p>
```
