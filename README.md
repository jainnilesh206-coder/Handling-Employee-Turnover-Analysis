# Employee Turnover Analysis Project

This repository section contains a complete machine learning project focused on predicting employee turnover, understanding the key drivers of attrition, clustering employees who left, handling class imbalance, comparing multiple classification models, and designing retention strategies based on predicted risk probabilities. The project is built from a formal problem statement, a structured HR dataset, and a full Python notebook implementation. [1][2][3]

## Project Overview

The project is framed around an HR analytics use case in which historical employee data is used to predict whether an employee will leave the company. The required workflow includes data quality checks, exploratory analysis, factor analysis, clustering of employees who left, SMOTE-based class balancing, 5-fold cross-validation, model comparison using ROC-AUC and confusion matrices, and risk-zone-based retention planning. [1]

The Python notebook implements this end-to-end workflow using pandas, NumPy, seaborn, matplotlib, scikit-learn, and imbalanced-learn. It combines business understanding with practical machine learning steps, making it a strong applied ML project rather than only a coding exercise. [3]

## Business Problem

The project aims to help HR teams identify employees who are likely to leave so that targeted retention actions can be planned in advance. The original problem statement defines employee turnover as the number of workers who leave the organization over time and asks for both predictive modeling and business recommendations. [1]

The project specifically tries to answer these questions:
- Are there missing-value or quality issues in the HR data? [1][3]
- Which features contribute most to employee turnover? [1][3]
- Can employees who left be grouped into behavioral clusters? [1][3]
- How should class imbalance be handled before training models? [1][3]
- Which supervised model performs best for turnover prediction? [1][3]
- Which retention actions should be taken for different turnover-risk levels? [1][3]

## Dataset Summary

The HR dataset used in the project contains employee-level records with numeric and categorical features related to satisfaction, performance, workload, tenure, accidents, promotions, department, salary, and target turnover label. The notebook reports 14,999 rows and 10 columns, with the target column `left` indicating whether an employee stayed (`0`) or left (`1`). [1][3]

The project statement defines the variables as follows: `satisfactionlevel`, `lastevaluation`, `numberproject`, `averagemontlyhours`, `timespendcompany`, `Workaccident`, `left`, `promotionlast5years`, `Department`, and `salary`. These columns are also visible in the notebook preview and dataset extracts. [1][2][3]

## Data Quality Check

The first stage of the notebook verifies whether missing values exist in the dataset. Using `data.info()` and explicit null checking, the notebook concludes that all 14,999 entries are non-null across all 10 columns and that no missing values are observed. [3]

This is an important early learning point from the project: before modeling begins, the dataset structure, dtypes, row counts, and null-status are validated. That shows good Python workflow discipline and aligns directly with the first requirement in the problem statement. [1][3]

## Exploratory Analysis

The notebook performs exploratory analysis to understand which features are associated with employee turnover. It generates a correlation heatmap for numerical features, distribution plots for satisfaction, last evaluation, and average monthly hours, and a bar plot comparing project counts for employees who stayed versus those who left. [1][3]

The notebook also records qualitative inferences from the project-count analysis, including observations that employees with certain project loads, especially around five projects, appear to have higher chances of leaving. It further notes that employees leaving the organization can appear across multiple profile types rather than one single fixed group. [3]

## Clustering of Employees Who Left

One of the project requirements is unsupervised clustering of employees who left based on satisfaction and evaluation. The notebook filters employees with `left = 1`, keeps `satisfactionlevel` and `lastevaluation`, and applies K-Means with `n_clusters=3`. [1][3]

The notebook then interprets the three clusters as broad behavioral categories:
- Low satisfaction with high evaluation. [3]
- Moderate satisfaction and moderate evaluation. [3]
- High satisfaction with high evaluation. [3]

This part of the project is valuable because it adds unsupervised learning inside a larger supervised problem. It shows that the project is not only about prediction accuracy, but also about discovering hidden employee patterns among attrition cases. [1][3]

## Class Imbalance Handling

The problem statement explicitly requires handling class imbalance using SMOTE. The notebook first separates numeric and categorical columns, converts categorical features into dummy variables using `pd.get_dummies()`, combines them into a final model-ready table, performs stratified train-test split with `test_size=0.2` and `random_state=123`, and then applies `SMOTE` on the training set. [1][3]

The notebook shows the training-label distribution before and after SMOTE. Before balancing, the training set contains 9,142 employees in class `0` and 2,857 in class `1`; after SMOTE, both classes become 9,142, producing a balanced training dataset. [3]

This stage demonstrates an important Python and machine learning learning outcome: imbalance should be handled after train-test splitting and on the training data only, not on the full dataset. The implementation reflects correct applied ML practice. [3]

## Model Training and Validation

The supervised modeling section trains three different classifiers on the SMOTE-balanced training data using 5-fold stratified cross-validation:
- Logistic Regression. [1][3]
- Random Forest Classifier. [1][3]
- Gradient Boosting Classifier. [1][3]

The notebook uses `StratifiedKFold` and `cross_val_predict` to generate out-of-fold predictions and then prints classification reports for each model. This is a strong learning pattern because it moves beyond one train-test result and evaluates model behavior more robustly across multiple folds. [1][3]

## Cross-Validation Results

The 5-fold CV classification report for Logistic Regression shows approximately 0.795 accuracy, with class-0 precision 0.810, class-0 recall 0.771, class-1 precision 0.781, and class-1 recall 0.819. This makes logistic regression a reasonable baseline but clearly weaker than the tree-based ensembles used later. [3]

The Random Forest 5-fold CV report shows about 0.985 accuracy, with both precision and recall near 0.975 to 0.995 across classes. The Gradient Boosting 5-fold CV report shows about 0.97 accuracy with similarly strong balance between precision and recall, though still slightly below Random Forest. [3]

## ROC-AUC Comparison

The notebook compares models using ROC-AUC on the original test set probabilities. The reported AUC values are 0.815 for Logistic Regression, 0.994 for Random Forest, and 0.989 for Gradient Boosting. [3]

Based on these scores, the notebook identifies Random Forest as the best-performing model overall because it has the highest ability to distinguish employees who leave from those who stay. This conclusion is explicitly stated in the notebook. [3]

## Confusion Matrix Analysis

The notebook also evaluates the three models using confusion matrices on the test set. The reported matrices are:

| Model | Confusion Matrix | Interpretation |
|---|---|---|
| Logistic Regression | `[[1769, 517], [184, 530]]` [3] | Higher false positives and false negatives than the ensemble models. [3] |
| Random Forest | `[[2270, 16], [16, 698]]` [3] | Very strong classification with low false positives and low false negatives. [3] |
| Gradient Boosting | `[[2234, 52], [49, 665]]` [3] | Strong performance, but slightly weaker than Random Forest. [3] |

The notebook argues that recall is the more important metric in this business context because false negatives are more costly. A false negative means the model predicts an employee will stay when the employee actually leaves, which directly harms retention planning. [3]

## Retention Strategy Logic

Using the best model, the notebook predicts turnover probabilities for the test set and divides employees into four risk zones based on score ranges. The defined zones are Safe Zone (Green) for probabilities below 0.20, Low-Risk Zone (Yellow) for 0.20 to below 0.60, Medium-Risk Zone (Orange) for 0.60 to below 0.90, and High-Risk Zone (Red) for 0.90 and above. [1][3]

The notebook prints the zone counts and reports 2,173 employees in the Safe Zone, 122 in the Low-Risk Zone, 53 in the Medium-Risk Zone, and 652 in the High-Risk Zone within the evaluated test-result frame. This makes the project useful not only for model benchmarking but also for downstream HR decision support. [3]

## Python Learning Inside This Project

This project contains multiple Python learning layers beyond the final model output. It is especially useful as a learning artifact because it shows how a real ML notebook is structured from import statements to business recommendations. [3]

### 1. Python Libraries and Imports

The notebook uses several important Python libraries for data science and machine learning:
- `numpy` for numeric handling. [3]
- `pandas` for data loading, filtering, preprocessing, and dataframe operations. [3]
- `matplotlib.pyplot` and `seaborn` for visual analysis. [3]
- `sklearn.cluster.KMeans` for clustering. [3]
- `sklearn.model_selection` tools such as `train_test_split`, `StratifiedKFold`, and `cross_val_predict`. [3]
- `sklearn.linear_model.LogisticRegression`. [3]
- `sklearn.ensemble.RandomForestClassifier` and `GradientBoostingClassifier`. [3]
- `sklearn.metrics` tools such as classification report, confusion matrix, ROC curve, and ROC-AUC score. [3]
- `imblearn.over_sampling.SMOTE` for imbalance correction. [3]

### 2. DataFrame Operations

The project teaches practical pandas operations such as:
- Reading CSV files with `pd.read_csv()`. [3]
- Inspecting head rows and schema. [3]
- Selecting numeric and categorical columns using `select_dtypes()`. [3]
- Dropping columns. [3]
- Combining dataframes using `pd.concat()`. [3]
- Creating result tables for predicted probabilities and risk labels. [3]

### 3. Preprocessing Skills

The project demonstrates several preprocessing concepts in Python:
- Separating target and feature variables. [3]
- Converting categorical variables into numeric features with `pd.get_dummies()`. [3]
- Building a final model-ready feature matrix. [3]
- Using stratified splitting to preserve target balance in train and test datasets. [3]
- Applying SMOTE only to training data. [3]

### 4. Visualization Skills

The notebook includes Python-based EDA and model-understanding visualizations such as:
- Correlation heatmaps. [1][3]
- Distribution plots. [1][3]
- Bar plots for comparing turnover across project counts. [1][3]
- Scatter-based cluster visualization after K-Means. [3]
- ROC curve analysis for model comparison. [3]

### 5. Unsupervised Learning Skills

The K-Means section teaches how to:
- Filter a subgroup of interest. [3]
- Select relevant clustering features. [1][3]
- Fit a `KMeans` model. [3]
- Retrieve cluster labels. [3]
- Plot clusters visually and write business-level cluster interpretations. [3]

### 6. Supervised Learning Skills

The model-training section teaches:
- Binary classification problem formulation. [1][3]
- Baseline vs advanced-model comparison. [3]
- How to use Logistic Regression, Random Forest, and Gradient Boosting in Python. [3]
- How to train models on balanced data. [3]
- How to evaluate models with cross-validation instead of relying on one split. [1][3]

### 7. Evaluation and Model Selection Skills

The project develops strong evaluation understanding through:
- Classification reports with precision, recall, F1-score, and support. [3]
- Confusion-matrix interpretation. [1][3]
- ROC and ROC-AUC comparison. [1][3]
- Choosing the best metric based on business cost, especially prioritizing recall because missed attrition cases are more harmful than extra alerts. [3]

### 8. Business Translation Skills

A particularly useful learning aspect is that the notebook does not stop at model accuracy. It converts output probabilities into HR-friendly risk zones, which is an example of turning machine learning predictions into actionable business categories. [1][3]

This demonstrates an important professional skill: model results must often be translated into operational decision frameworks rather than only reported as metrics. [1][3]

## Main Project Workflow

The Python notebook follows this practical workflow:
1. Import libraries and load the HR dataset. [3]
2. Inspect the data and verify there are no missing values. [3]
3. Explore important variables using plots and correlation analysis. [1][3]
4. Cluster employees who left using K-Means. [1][3]
5. Convert categorical features into numeric form. [1][3]
6. Split the data with stratification. [1][3]
7. Apply SMOTE on the training data. [1][3]
8. Train Logistic Regression, Random Forest, and Gradient Boosting models using 5-fold CV. [1][3]
9. Compare models using classification report, ROC-AUC, and confusion matrix. [1][3]
10. Select the best model and create risk-zone-based retention outputs. [1][3]

## Key Results

The major findings from the implemented notebook are:
- No missing values were found in the dataset. [3]
- Employees who left can be explored through satisfaction, evaluation, and project-load patterns. [1][3]
- K-Means on employees who left forms three broad satisfaction-evaluation clusters. [3]
- SMOTE successfully balances the minority attrition class in training data from 2,857 to 9,142. [3]
- Random Forest is the best-performing model with AUC 0.994 and a very strong confusion matrix. [3]
- Recall is treated as especially important because missing a true attrition case is more costly than raising an extra retention alert. [3]
- The final output supports risk-zone-based employee retention planning. [1][3]


## Why This Project Matters

This project is a strong portfolio example because it combines data cleaning, visualization, clustering, imbalance correction, supervised learning, cross-validation, metric-driven model selection, and business interpretation in one notebook. It shows both machine learning understanding and applied Python execution. [1][3]

It is also valuable from a learning perspective because it demonstrates how a real-world ML problem is solved in stages: understand the problem, inspect the data, explore patterns, prepare features, handle imbalance, train several models, compare them fairly, and translate the results into business action. [1][3]
