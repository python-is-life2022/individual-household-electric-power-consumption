# Individual Household Electric Power Consumption

## Project Overview

This project focuses on analyzing household electricity consumption data and developing a deep learning model for forecasting future energy usage.

The project is currently in progress and is being developed step by step, from exploratory data analysis and preprocessing to time-series modeling and deep learning.

## Objectives

- Perform exploratory data analysis (EDA)
- Identify missing values and potential outliers
- Analyze feature distributions and relationships
- Visualize electricity consumption over time
- Identify temporal patterns and trends
- Prepare the dataset for time-series forecasting
- Perform temporal feature engineering
- Apply appropriate feature scaling and preprocessing techniques
- Build deep learning models using TensorFlow/Keras
- Evaluate and compare forecasting approaches

## Dataset

The dataset used in this project is the **Individual Household Electric Power Consumption** dataset from the UCI Machine Learning Repository.

The dataset contains nearly four years of one-minute measurements of household electricity consumption.

It includes the following features:

- `Date`
- `Time`
- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

The UCI dataset does not define a specific target variable. The final target and forecasting task will be defined based on the final machine learning problem.

## Libraries and Tools

- **Python** — Programming language
- **Jupyter Notebook** — Development and analysis environment
- **Pandas** — Data loading, manipulation, and analysis
- **Matplotlib** — Data visualization
- **Scikit-learn** — Data splitting, preprocessing, and model selection utilities
- **Seaborn** — Statistical visualization
- **TensorFlow / Keras** — Deep learning and time-series modeling

## Project Workflow

### Stage 1 — Data Loading

The UCI Individual Household Electric Power Consumption dataset was loaded into a Pandas DataFrame.

### Stage 2 — Initial Data Inspection

The dataset was inspected to understand its structure and data quality.

The following steps were performed:

- Checked dataset dimensions
- Inspected column names
- Checked data types
- Calculated descriptive statistics
- Identified missing values
- Examined unique values and value frequencies

### Stage 3 — Exploratory Data Analysis

Exploratory data analysis was performed to better understand the distributions and relationships between the numerical features.

The following analyses were performed:

- Feature distributions
- Skewness analysis
- Potential outlier detection using the IQR method
- Correlation analysis
- Correlation matrix visualization

Potential outliers detected using the IQR method:

- `Global_active_power`: 94,907
- `Voltage`: 51,067

Feature skewness:

- `Global_active_power`: ~1.79
- `Global_reactive_power`: ~1.26
- `Voltage`: ~-0.33
- `Global_intensity`: ~1.85
- `Sub_metering_1`: ~5.94
- `Sub_metering_2`: ~7.09
- `Sub_metering_3`: ~0.72

Correlation with `Global_active_power`:

- `Global_intensity`: ~1.00
- `Sub_metering_3`: ~0.64
- `Sub_metering_1`: ~0.48
- `Sub_metering_2`: ~0.44
- `Global_reactive_power`: ~0.25
- `Voltage`: ~-0.40

The correlation matrix was saved as:

`images/correlation_matrix.png`

Additional EDA will focus on visualizing consumption over time and identifying temporal patterns.

### Stage 4 — Missing Value Handling

Missing values in the numerical features were handled using statistical imputation.

Because several electricity consumption features showed positive skewness, median imputation was used for:

- `Global_active_power`
- `Global_reactive_power`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

Mean imputation was used for `Voltage` because its distribution was approximately symmetric with only slight negative skewness.

After imputation, all numerical features contained 0% missing values.

### Stage 5 — Temporal Feature Engineering and Chronological Split

The `Date` and `Time` columns were combined into a single `datetime` column using the format:

`%d/%m/%Y %H:%M:%S`

Several temporal features were extracted:

- `Year`
- `Month`
- `Day`
- `DayOfWeek`
- `Hour`
- `Minute`

`DayOfWeek` represents the day of the week using values from 0 (Monday) to 6 (Sunday).

The original `datetime` column was kept for further temporal analysis and feature engineering.

The `model_selection` module from Scikit-learn was used to split the dataset chronologically.

`train_test_split` was used with `shuffle=False` to preserve the temporal order of the observations.

The dataset was divided into:

- **Training Set:** 72%
- **Validation Set:** 8%
- **Test Set:** 20%

Resulting dataset sizes:

- Training: **1,494,186 observations**
- Validation: **166,021 observations**
- Test: **415,052 observations**

The resulting structure is:

**Past → Training → Validation → Test → Future**

This chronological split helps prevent future information from being introduced into the training process.

### Stage 6 — Feature Scaling

The numerical electricity measurements were standardized using `StandardScaler` from Scikit-learn.

The scaler was fitted only on the training set. The same fitted scaler was then used to transform the validation and test sets.

This prevents data leakage because the mean and standard deviation used for scaling are calculated only from the training data.

The preprocessing workflow is:

- **Training Set:** `fit_transform()`
- **Validation Set:** `transform()`
- **Test Set:** `transform()`

The standardized numerical features include:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`

Temporal features will be handled separately. Features such as `Hour`, `DayOfWeek`, and `Month` can later be represented using cyclical encoding to preserve their periodic nature.

## Experiment Log

### Completed

- Loaded the UCI electricity consumption dataset
- Inspected dataset structure and data types
- Analyzed missing values
- Calculated descriptive statistics
- Performed distribution and skewness analysis
- Identified potential outliers using the IQR method
- Performed correlation analysis
- Created a correlation matrix visualization
- Handled missing values using statistical imputation
- Combined `Date` and `Time` into a `datetime` column
- Extracted temporal features
- Created chronological training, validation, and test sets
- Applied `StandardScaler` to numerical features using training data only

### Next Steps

- Complete time-series EDA
- Visualize daily, monthly, and seasonal consumption patterns
- Create additional temporal features
- Apply cyclical encoding to appropriate time features
- Define the final forecasting target
- Determine an appropriate sequence length
- Prepare input-output sequences
- Apply additional preprocessing if necessary
- Build deep learning forecasting models using TensorFlow/Keras
- Evaluate model performance on the test set

## Project Status

**Current Stage:** Data preprocessing and feature engineering

### Completed

- Data loading and inspection
- Exploratory data analysis
- Missing value handling
- Temporal feature engineering
- Chronological dataset splitting
- Numerical feature scaling

### In Progress

- Time-series pattern analysis
- Temporal feature optimization
- Final definition of the forecasting problem

### Planned

- Sequence generation
- Deep learning model development
- Model training and validation
- Final evaluation and comparison
