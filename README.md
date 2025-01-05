# Aeropolis 🚁

**Team Members:** 
- Maria Maggiora (297281)
- Alessia Maria Cercaci (299981)

## Introduction 🖍️

In the futuristic city of Aeropolis, autonomous delivery drones are revolutionizing the way goods are transported across the sprawling metropolis, ensuring fast and efficient delivery. These drones play a pivotal role in maintaining the city's dynamic pace, with their performance evaluated by the amount of cargo they can deliver per flight. However, optimizing drone performance is no simple task, as it depends on a multitude of factors, including weather conditions, flight altitude, terrain type, and other operational variables.

Therefore our project addresses the challenge of optimizing drone logistics in Aeropolis by predicting cargo capacity under diverse conditions, thereby improving delivery efficiency and resource utilization. By analyzing a rich dataset encompassing 20 variables, we seek to enhance drone performance, optimize resource allocation, and contribute to the development of smarter urban logistics solutions. 

Our approach is built around a structured data science workflow. We begin with an in-depth Exploratory Data Analysis (EDA) to uncover insights within the dataset. This phase includes visualizing trends, detecting anomalies, and identifying the most critical factors influencing drone performance. The insights gained here serve as the foundation for all subsequent steps in the analysis.

Next, we move to the data preprocessing phase, where the raw dataset is transformed into a format suitable for machine learning models. This includes handling missing values, scaling numerical variables, and encoding categorical features. These tailored preprocessing steps not only prepare the data for analysis but also help highlight hidden relationships that could enhance the predictive power of our models.

Finally, we embark on a detailed experimentation process, testing various machine learning algorithms to identify the most effective approach for predicting drone cargo capacity. Each model is carefully fine-tuned, and its performance is evaluated to ensure it meets the high standards required for Aeropolis’s dynamic delivery ecosystem. This iterative process ensures that our final model is both robust and accurate, ready to contribute to the optimization of Aeropolis’s autonomous delivery systems.

The results will enable Aeropolis to continue leading the way in futuristic logistics and urban innovation! 🥳

------------

## Methods 🔎

Our project began with a thorough analysis of the Aeropolis dataset, employing Python and its powerful ecosystem of libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn. This exploratory phase was essential for understanding the dataset’s unique characteristics, including the distribution of key variables and its structure. It has permitted to laid a strong foundation for predictive modeling.

### Dataset Analysis
The EDA started with a thorough examination of the dataset, focusing on understanding its structure, the distribution of variables, checked for anomalies, and any initial trends or patterns.

The Aeropolis dataset comprises 20 features that span environmental, operational, and logistical factors. Below is a preview of the dataset used in this project:

![Dataset Preview](images/dataset_preview.png "Dataset Preview")

Therefore it consists of:
- **Target Variable**: `Cargo_Capacity_kg` (continuous).
- **Features**: Numerical (e.g., `Flight_Hours`, `Vertical_Max_Speed`) and categorical (e.g., `Weather_Status`, `Package_Type`).

We've also focused on understanding the distribution of both numerical and categorical variables. These univariate analyses provided valuable insights, revealing multimodal distributions in some variables and skewed distributions in others. Additionally, this analysis informed our strategy for imputing missing values later in the data preprocessing pipeline.

# METTRE IMMAGINE

### Identification of Anomalies
During the initial Exploratory Data Analysis (EDA), we identified several anomalies within the dataset. 

One notable issue, as we've prevusly said, was that the feature Cleaning_Liquid_Usage_liters exhibited a significant right skew. This skewness led to an imbalance in the dataset, adversely impacting the performance of our models.

Additionally, our analysis revealed that features such as Cargo_Capacity_kg, Route_Optimization_Per_Second, and Water_Usage_liters contained negative values. These are logically implausible, as such variables can't assume negative values in real-world scenarios. To address this inconsistency, we decided to replace these erroneous values with NaN. This approach aligns with our data preprocessing pipeline, which includes systematic handling of missing values later in the workflow.

Lastly, the column Flight_Duration_Minutes was found to contain Boolean values, which is clearly incorrect since it should represent numerical data. Given that this column was not providing meaningful insights and was irreparably flawed, we decided to remove it from the dataset entirely.

### Filling Missing Data
We then checked for missing values in the dataset and visualized the results to conduct a thorough analysis. Our findings revealed that nearly 10% of all features contained missing values, which could pose significant challenges during model training.
# IMMAGE
At the beginning of our preprocessing, we attempted to fill missing values using the K-Nearest Neighbors (KNN) imputation method. While this algorithm is effective in many scenarios, it proved to be too slow for our dataset due to the large volume of missing values and the associated computational complexity.
Faced with this challenge, we sought an alternative solution. After examining the correlation of missing values across features, we concluded that the missing data appeared independently in each column. This observation indicates that the dataset follows a Missing Completely at Random (MCAR) pattern.
# IMMAGE
Based on this analysis, we decided to handle missing values by imputing numerical columns with their median and categorical columns with their mode. However, for the Cargo_Capacity_Column, which serves as the target variable, we opted not to apply imputation, as maintaining accuracy in this column is critical, we've just decided to erase all the missing values of this column.

### Preprocessing
**Preprocessing**

**Feature Encoding**

To prepare the dataset for analysis, categorical variables such as **Weather_Status** and **Terrain_Type** were transformed using one-hot encoding. This technique created binary columns for each unique category, ensuring compatibility with machine learning models. By using `pd.get_dummies()` with `drop_first=True`, we avoided multicollinearity issues stemming from the dummy variable trap. Binary features like True/False values were similarly encoded as 0s and 1s, facilitating numerical processing. The final dataset contained 34 features, ready for further analysis.
![relationmatrix](![Unknown](https://github.com/user-attachments/assets/790af350-a0c2-4575-845a-368a591f33a7))


**Correlation Analysis**

Two correlation matrices were computed to identify relationships between features and the target variable, `Cargo_Capacity_kg`. The first matrix included all features, revealing the following key insights:

- **Wind_Speed_kmph** exhibited a moderate positive correlation (0.73) with `Cargo_Capacity_kg`, suggesting higher wind speeds lead to greater cargo capacity.
- **Quantum_Battery_True** showed a moderate positive correlation (0.40) with Cargo_Capacity_kg`, indicating that drones equipped with quantum batteries tend to carry more cargo.
- **Air_Temperature_Celsius** had a weak positive correlation (0.08) with the target variable.

The second heatmap focused only on the most relevant features, highlighting those with stronger relationships to the target. Features with negligible correlations, such as **Route_Optimization_Per_Second** and **Flight_Zone_North**, were marked for potential removal but since we aren't in a linear relationship correlation, we must use other techinques to analyze the relationship between the values, and calculate how important each feature is in predicting Cargo_Capacity_kg.
This process ensured that subsequent modeling would focus on impactful predictors, improving model efficiency. 

**Feature Selection via Mutual Information**

To complement correlation analysis, we applied Mutual Information (MI) to evaluate feature importance. Unlike correlation, MI measures both linear and non-linear dependencies, offering a holistic view of feature relevance.

Features with MI scores of 0 were deemed irrelevant and removed. This step ensured that the dataset was streamlined, retaining only impactful variables. For example, **Wind_Speed_kmph**, **Quantum_Battery_True**, and **Air_Temperature_Celsius** emerged as key predictors of `Cargo_Capacity_kg`. By removing irrelevant features, we reduced noise and improved computational efficiency.

**Objective of Feature Selection**

The primary goal of this process was to refine the dataset by eliminating features with negligible or no predictive power. This ensures a cleaner dataset, reduces overfitting risks, and allows models to focus on the most relevant variables, enhancing both interpretability and performance.

**Dataset Splitting, Scaling, and Reflection**

After feature selection, the dataset was divided into training and test subsets using an 80/20 split, this is important to avoid overfitting. Standard scaling was applied to standardize the magnitude of features, ensuring uniformity and compatibility with machine learning algorithms.

### Model Selection and Rationale

Given the regression nature of the problem, we selected:
- **Linear Regression**: Baseline model to set benchmarks.
- **Random Forest Regressor**: Captures non-linear relationships and handles high-dimensional data effectively.
- **Hist Gradient Boosting (Tuned)**: Achieved strong predictive performance through hyperparameter tuning.

We utilized **Python** and libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn. Our environment configuration is included in the `environment.yml` file.



### Key Steps in Methodology
1. **Data Acquisition and Inspection:**
   - The Aeropolis dataset was imported, inspected for completeness, and checked for anomalies. A comprehensive overview of its 20 features revealed relationships that could influence cargo capacity predictions. 
2. **Data Cleaning:**
   - Missing values were addressed through imputation methods tailored to the feature type (e.g., mean for numerical variables and mode for categorical variables). Outliers were removed based on domain knowledge and statistical thresholds to ensure data quality.
3. **Exploratory Data Analysis (EDA):** --> cambiare tittolo 
   - Visualizations, including heatmaps, histograms, and scatter plots, were created to explore correlations between features and cargo capacity. These helped identify trends, relationships, and potential preprocessing needs.
   - Example visualizations include:
     - Heatmap of feature correlations:
       ![Heatmap](images/heatmap.png "Heatmap of Feature Correlations")
     - Distribution of Cargo Capacity:
       ![Cargo Capacity Distribution](images/cargo_distribution.png "Cargo Capacity Distribution")
4. **Feature Engineering:**
   - Features were scaled or transformed to ensure uniformity and enhance model performance.
   - Key variables such as wind speed and terrain type were emphasized due to their significant impact on drone operations.
   - Categorical features, like package type and weather status, were one-hot encoded to enable compatibility with machine learning models.
5. **Model Selection and Training:**
   - Multiple models, including Linear Regression, Random Forest, and Gradient Boosting, were tested to determine the most suitable approach for predicting cargo capacity. Models were selected based on their ability to handle numerical and categorical data, scalability, and interpretability.
6. **Optimization:**
   - Hyperparameter tuning was performed using Grid Search Cross-Validation to refine model accuracy and efficiency. Specific parameters such as maximum depth, learning rate, and number of estimators were optimized for each model.

### Workflow Visualization
The project workflow can be summarized as follows:

```markdown
1. **Data Loading:** Import and inspect the Aeropolis dataset.
2. **EDA:** Perform exploratory data analysis with visualizations.
3. **Preprocessing:** Handle missing data, remove outliers, and encode features.
4. **Modeling:** Train and evaluate machine learning models.
5. **Optimization:** Hyperparameter tuning of the best-performing model.
6. **Results:** Analyze performance metrics and derive insights.
```

![Workflow Visualization](images/workflow.png "Workflow Visualization")

### Optimization and Computational Efficiency
The computational efficiency of the models was evaluated, and the time taken by each model is shown below:

![Time Complexity Comparison](images/time_complexity.png "Comparative Analysis of Time Complexity Across Models")

From the chart, it is evident that Linear Regression is the most efficient model in terms of computational time, followed by Hist Gradient Boosting and Random Forest. This highlights the suitability of Linear Regression for scenarios where speed is critical.

### Tools and Environment
- **Programming Language:** Python (v3.x)
- **Libraries Used:**
  - pandas (v1.x): Data manipulation and analysis
  - numpy (v1.x): Numerical computations
  - scikit-learn (v1.x): Machine learning algorithms and preprocessing
  - matplotlib (v3.x) and seaborn (v0.x): Data visualization
- **Environment Setup:**
  - To ensure reproducibility, the project environment can be recreated using the following command:
    ```
    conda env create -f environment.yml
    ```
- **Dataset:** The Aeropolis dataset (`aeropolis.csv`), containing variables such as Wind Speed, Air Temperature, and Cargo Capacity, forms the core of this analysis.

By following this systematic approach, we ensured a robust foundation for predictive modeling, enabling the extraction of actionable insights and the development of reliable machine learning models.

## [Section 3] Experimental Design 🔬

### Main Purpose
The primary goal of this project is to evaluate the performance of various machine learning models in predicting drone cargo capacity. Specific objectives include:
- Identifying the best-performing model in terms of accuracy and computational efficiency.
- Exploring the impact of preprocessing techniques on model performance.

### Baselines
The following models were used as baselines:
- **Linear Regression:** A simple and interpretable model ideal for understanding linear relationships.
- **Random Forest:** A tree-based ensemble method capable of capturing non-linear relationships and handling diverse data types.
- **Gradient Boosting:** A powerful boosting technique known for its efficiency in regression tasks.

### Evaluation Metrics
To assess model performance, we used:
- **Root Mean Squared Error (RMSE):** Highlights large prediction errors and emphasizes the magnitude of error.
- **R-squared (R²):** Measures the proportion of variance in the target variable explained by the model, indicating its explanatory power.
- **MAE** 

### Experimental Details
- **Data Partitioning:** The dataset was split into training (80%) and test (20%) subsets.
- **Hyperparameter Tuning:** Performed using Grid Search Cross-Validation for Gradient Boosting. Key parameters included:
  - Learning Rate
  - Maximum Depth
  - Number of Estimators

By comparing baseline performances and applying optimization techniques, we aimed to achieve the best possible predictive performance for the models.

## [Section 4] Results 🏆

### Main Findings
1. **Linear Regression** achieved a balanced trade-off between simplicity, computational efficiency, and predictive performance.
2. While **Gradient Boosting** and **Random Forest** demonstrated slightly higher train-set performance, Linear Regression performed equivalently or better on the test set, ensuring its robustness against overfitting.

### Visualizations and Tables
- **Feature Correlations Heatmap:** Visualizing relationships between key features:
  ![Feature Correlations Heatmap](images/heatmap.png "Feature Correlations Heatmap")

- **Model Performance Comparison:** Bar chart showing RMSE and R² for each model:
  ![Model Performance Comparison](images/model_metrics.png "Model Evaluation Metrics Comparison")

### Model Performance
The comparison of regression models reveals the following performance metrics:

| Model                   | MAE (Test) | RMSE (Test) | R² (Test) |
|-------------------------|-------------|--------------|------------|
| Linear Regression       | 0.7535     | 0.9392      | 0.6937    |
| Random Forest           | 0.7651     | 0.9566      | 0.6822    |
| Random Forest Tuned     | 0.7778     | 0.9723      | 0.6717    |
| Hist Gradient Boosting  | 0.7538     | 0.9396      | 0.6934    |
| Hist Gradient Boosting Tuned | 0.7538     | 0.9396      | 0.6934    |


### Insights
Linear Regression emerged as the best model due to its simplicity, computational efficiency, and comparable accuracy to more complex models. It is especially suitable for scenarios requiring interpretable models and faster computations.

## [Section 5] Conclusions 🖋️

### Summary
This study demonstrates the feasibility of leveraging machine learning to predict drone cargo capacity in Aeropolis. By utilizing models such as Linear Regression, we effectively identified critical factors influencing drone operations and optimized predictive accuracy. The results highlight the importance of preprocessing and feature selection in achieving robust model performance.

### Future Directions
Future research could include:
- Integrating real-time drone telemetry data for dynamic predictions.
- Exploring deep learning models for capturing higher-dimensional patterns.
- Evaluating the impact of additional environmental variables, such as precipitation or wind gust patterns.

By expanding the scope of data and methodologies, this project could further enhance drone logistics in Aeropolis and similar urban environments.


[04.01.25 20:18:14] Maria: Okok
[04.01.25 20:18:21] Ale 💂🏻🧝🏻‍♀️🪖: # Aeropolis 🚁

**Team Members:** 
- Maria Maggiora (297281)
- Alessia Maria Cercaci (299981)

## Introduction 🖍️

In the futuristic city of Aeropolis, autonomous delivery drones are revolutionizing the way goods are transported across the sprawling metropolis, ensuring fast and efficient delivery. These drones play a pivotal role in maintaining the city's dynamic pace, with their performance evaluated by the amount of cargo they can deliver per flight. However, optimizing drone performance is no simple task, as it depends on a multitude of factors, including weather conditions, flight altitude, terrain type, and other operational variables.

Therefore our project addresses the challenge of optimizing drone logistics in Aeropolis by predicting cargo capacity under diverse conditions, thereby improving delivery efficiency and resource utilization. By analyzing a rich dataset encompassing 20 variables, we seek to enhance drone performance, optimize resource allocation, and contribute to the development of smarter urban logistics solutions. 

Our approach is built around a structured data science workflow. We begin with an in-depth Exploratory Data Analysis (EDA) to uncover insights within the dataset. This phase includes visualizing trends, detecting anomalies, and identifying the most critical factors influencing drone performance. The insights gained here serve as the foundation for all subsequent steps in the analysis.

Next, we move to the data preprocessing phase, where the raw dataset is transformed into a format suitable for machine learning models. This includes handling missing values, scaling numerical variables, and encoding categorical features. These tailored preprocessing steps not only prepare the data for analysis but also help highlight hidden relationships that could enhance the predictive power of our models.

Finally, we embark on a detailed experimentation process, testing various machine learning algorithms to identify the most effective approach for predicting drone cargo capacity. Each model is carefully fine-tuned, and its performance is evaluated to ensure it meets the high standards required for Aeropolis’s dynamic delivery ecosystem. This iterative process ensures that our final model is both robust and accurate, ready to contribute to the optimization of Aeropolis’s autonomous delivery systems.

The results will enable Aeropolis to continue leading the way in futuristic logistics and urban innovation! 🥳

 
## Methods 🔍

Our project methodology included detailed steps to explore, preprocess, and model the dataset effectively:

### Dataset Examination

The dataset consists of:
- **Target Variable**: `Cargo_Capacity_kg` (continuous).
- **Features**: Numerical (e.g., `Flight_Hours`, `Vertical_Max_Speed`) and categorical (e.g., `Weather_Status`, `Package_Type`).

#### Key Observations:
1. **Outliers**: Extreme values identified in `Cargo_Capacity_kg` and `Flight_Hours` using boxplots and z-scores.
2. **Missing Data**: Categorical features like `Weather_Status` had missing values that needed imputation.
3. **Feature Correlations**: Heatmaps revealed strong correlations between `Flight_Hours` and `Cargo_Capacity_kg`.
4. **Skewed Distributions**: Some numerical features displayed skewness, requiring transformation for better model performance.

### Identification of Anomalies

During EDA, we found:
- Inconsistent values in `Wind_Speed_kmph`, which were addressed through domain-driven thresholding.
- High variability in `Cargo_Capacity_kg` for specific `Terrain_Type` categories, requiring stratified analysis.

### Preprocessing

The preprocessing steps included:
1. **Outlier Removal**:
   - Removed extreme values in `Cargo_Capacity_kg` and `Flight_Duration_Minutes` based on interquartile ranges.
2. **Missing Value Imputation**:
   - Used mode for categorical variables like `Package_Type` and median for numerical ones.
3. **Feature Encoding**:
   - Applied one-hot encoding for categorical features such as `Weather_Status` and `Terrain_Type`.
4. **Feature Scaling**:
   - Standardized numerical variables to ensure compatibility with regression algorithms.

### Dataset Splitting
- **Training Set**: 80%
- **Testing Set**: 20%
- **Cross-validation**: Used 5-fold cross-validation for model evaluation.

### Model Selection and Rationale

Given the regression nature of the problem, we selected:
- **Linear Regression**: Baseline model to set benchmarks.
- **Random Forest Regressor**: Captures non-linear relationships and handles high-dimensional data effectively.
- **Hist Gradient Boosting (Tuned)**: Achieved strong predictive performance through hyperparameter tuning.

We utilized **Python** and libraries such as Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn. Our environment configuration is included in the `environment.yml` file.

---

### MANCA FLOWCHART DEL WORKFLOW

```
![Workflow Diagram](images/workflow_diagram.png)
```

---

## Experimental Design 🔬

The project was divided into two phases to assess the impact of preprocessing and model selection:

### Phase 1: Baseline Model Evaluation
- **Objective**: Establish initial benchmarks without preprocessing.
- **Models Tested**:
  - Linear Regression
  - Random Forest
  - Hist Gradient Boosting
- **Findings**: Linear Regression yielded promising results with the lowest MAE and RMSE, making it a strong candidate for deployment.

### Phase 2: Enhanced Model Evaluation
- **Objective**: Assess the impact of preprocessing and advanced tuning on model performance.
- **Adjustments**:
  1. Removed outliers and transformed skewed distributions.
  2. Applied hyperparameter tuning using GridSearchCV.
- **Models Tested**:
  - Linear Regression
  - Random Forest (Tuned)
  - Hist Gradient Boosting (Tuned)
- **Results**: Linear Regression remained the most accurate model with:
  - **MAE (Train)**: 0.7552
  - **RMSE (Train)**: 0.9399
  - **R² (Train)**: 0.6923
  - **MAE (Test)**: 0.7535
  - **RMSE (Test)**: 0.9392
  - **R² (Test)**: 0.6937

---

### MANCA IMMAGINE DELLA PERFORMANCE DEI MODELS

```
![Model Performance Comparison](images/model_performance.png)
```

---

## Results 🏅

### EDA Highlights
1. **Key Correlations**:
   - Features like `Flight_Hours`, `Weather_Status`, and `Route_Optimization_Per_Second` were highly correlated with `Cargo_Capacity_kg`, as visualized in the heatmap below:

```
![EDA Heatmap](images/eda_heatmap.png)
```

2. **Anomalies Detected**:
   - Outliers in `Wind_Speed_kmph` and `Cargo_Capacity_kg`.
   - Missing values in categorical features like `Weather_Status`, which were addressed during preprocessing.

### Model Performance
The comparison of regression models reveals the following performance metrics:

| Model                   | MAE (Train) | RMSE (Train) | R² (Train) | MAE (Test) | RMSE (Test) | R² (Test) |
|-------------------------|-------------|--------------|------------|------------|-------------|-----------|
| Linear Regression       | 0.7552      | 0.9399       | 0.6923     | 0.7535     | 0.9392      | 0.6937    |
| Random Forest           | 0.2838      | 0.3577       | 0.9554     | 0.7651     | 0.9566      | 0.6822    |
| Random Forest Tuned     | 0.7750      | 0.9677       | 0.6739     | 0.7778     | 0.9723      | 0.6717    |
| Hist Gradient Boosting  | 0.7541      | 0.9386       | 0.6932     | 0.7538     | 0.9396      | 0.6934    |
| Hist Gradient Boosting Tuned | 0.7541      | 0.9386       | 0.6932     | 0.7538     | 0.9396      | 0.6934    |

---

## Conclusions 🖋️

### Key Takeaways:
1. **Linear Regression Outperforms Other Models**: Across all metrics (MAE, RMSE, R²), Linear Regression consistently demonstrated the best performance, highlighting its suitability for this dataset.
2. **Hist Gradient Boosting is a Strong Alternative**: With comparable performance to Linear Regression, it showcases robustness and generalization capabilities.
3. **Random Forest Requires Further Tuning**: Its overfitting tendencies limit its generalizability to unseen data.

---

### MANCA IMMAGINE
