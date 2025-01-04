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
