Employee Turnover Analysis Project
This repository section contains a complete machine learning project focused on predicting employee turnover, understanding the key drivers of attrition, clustering employees who left, handling class imbalance, comparing multiple classification models, and designing retention strategies based on predicted risk probabilities. The project is built from a formal problem statement, a structured HR dataset, and a full Python notebook implementation.

Project Overview
The project is framed around an HR analytics use case in which historical employee data is used to predict whether an employee will leave the company. The required workflow includes data quality checks, exploratory analysis, factor analysis, clustering of employees who left, SMOTE-based class balancing, 5-fold cross-validation, model comparison using ROC-AUC and confusion matrices, and risk-zone-based retention planning.

The Python notebook implements this end-to-end workflow using pandas, NumPy, seaborn, matplotlib, scikit-learn, and imbalanced-learn. It combines business understanding with practical machine learning steps, making it a strong applied ML project rather than only a coding exercise.

Business Problem
The project aims to help HR teams identify employees who are likely to leave so that targeted retention actions can be planned in advance. The original problem statement defines employee turnover as the number of workers who leave the organization over time and asks for both predictive modeling and business recommendations.

The project specifically tries to answer these questions:

Are there missing-value or quality issues in the HR data?

Which features contribute most to employee turnover?

Can employees who left be grouped into behavioral clusters?

How should class imbalance be handled before training models?

Which supervised model performs best for turnover prediction?

Which retention actions should be taken for different turnover-risk levels?

Dataset Summary
The HR dataset used in the project contains employee-level records with numeric and categorical features related to satisfaction, performance, workload, tenure, accidents, promotions, department, salary, and the target turnover label. The notebook reports 14,999 rows and 10 columns, with the target column left indicating whether an employee stayed (0) or left (1).

The project statement defines the variables as follows: satisfactionlevel, lastevaluation, numberproject, averagemontlyhours, timespendcompany, Workaccident, left, promotionlast5years, Department, and salary. These columns are also visible in the notebook preview and dataset extracts.

Data Quality Check
The first stage of the notebook verifies whether missing values exist in the dataset. Using data.info() and explicit null checking, the notebook concludes that all 14,999 entries are non-null across all 10 columns and that no missing values are observed.

This is an important early learning point from the project: before modeling begins, the dataset structure, dtypes, row counts, and null status are validated. That shows good Python workflow discipline and aligns directly with the first requirement in the problem statement.

Exploratory Analysis
The notebook performs exploratory analysis to understand which features are associated with employee turnover. It generates a correlation heatmap for numerical features, distribution plots for satisfaction, last evaluation, and average monthly hours, and a bar plot comparing project counts for employees who stayed versus those who left.

The notebook also records qualitative inferences from the project-count analysis, including observations that employees with certain project loads, especially around five projects, appear to have higher chances of leaving. It further notes that employees leaving the organization can appear across multiple profile types rather than one single fixed group.

Clustering of Employees Who Left
One of the project requirements is unsupervised clustering of employees who left based on satisfaction and evaluation. The notebook filters employees with left = 1, keeps satisfactionlevel and lastevaluation, and applies K-Means with n_clusters=3.

The notebook then interprets the three clusters as broad behavioral categories:

Low satisfaction with high evaluation

Moderate satisfaction and moderate evaluation

High satisfaction with high evaluation

This part of the project is valuable because it adds unsupervised learning inside a larger supervised problem. It shows that the project is not only about prediction accuracy, but also about discovering hidden employee patterns among attrition cases.

Class Imbalance Handling
The problem statement explicitly requires handling class imbalance using SMOTE. The notebook first separates numeric and categorical columns, converts categorical features into dummy variables using pd.get_dummies(), combines them into a final model-ready table, performs stratified train-test split with test_size=0.2 and random_state=123, and then applies SMOTE on the training set.

The notebook shows the training-label distribution before and after SMOTE. Before balancing, the training set contains 9,142 employees in class 0 and 2,857 in class 1; after SMOTE, both classes become 9,142, producing a balanced training dataset.

This stage demonstrates an important Python and machine learning outcome: imbalance should be handled after train-test splitting and on the training data only, not on the full dataset. The implementation reflects correct applied ML practice.

Model Training and Validation
The supervised modeling section trains three different classifiers on the SMOTE-balanced training data using 5-fold stratified cross-validation:

Logistic Regression

Random Forest Classifier

Gradient Boosting Classifier

The notebook uses StratifiedKFold and cross_val_predict to generate out-of-fold predictions and then prints classification reports for each model. This is a strong learning pattern because it moves beyond one train-test result and evaluates model behavior more robustly across multiple folds.

Cross-Validation Results
The 5-fold CV classification report for Logistic Regression shows approximately 0.795 accuracy, with class-0 precision 0.810, class-0 recall 0.771, class-1 precision 0.781, and class-1 recall 0.819. This makes Logistic Regression a reasonable baseline but clearly weaker than the tree-based ensembles used later.

The Random Forest 5-fold CV report shows about 0.985 accuracy, with both precision and recall near 0.975 to 0.995 across classes. The Gradient Boosting 5-fold CV report shows about 0.97 accuracy with similarly strong balance between precision and recall, though still slightly below Random Forest.

ROC-AUC Comparison
The notebook compares models using ROC-AUC on the original test set probabilities. The reported AUC values are 0.815 for Logistic Regression, 0.994 for Random Forest, and 0.989 for Gradient Boosting.

Based on these scores, the notebook identifies Random Forest as the best-performing model overall because it has the highest ability to distinguish employees who leave from those who stay. This conclusion is explicitly stated in the notebook.

Confusion Matrix Analysis
The notebook also evaluates the three models using confusion matrices on the test set. The reported matrices are:

Model	Confusion Matrix	Interpretation
Logistic Regression	[[1769, 517], [184, 530]]	Higher false positives and false negatives than the ensemble models.
Random Forest	[[2270, 16], [16, 698]]	Very strong classification with low false positives and low false negatives.
Gradient Boosting	[[2234, 52], [49, 665]]	Strong performance, but slightly weaker than Random Forest.
The notebook argues that recall is the more important metric in this business context because false negatives are more costly. A false negative means the model predicts an employee will stay when the employee actually leaves, which directly harms retention planning.

Retention Strategy Logic
Using the best model, the notebook predicts turnover probabilities for the test set and divides employees into four risk zones based on score ranges. The defined zones are Safe Zone (Green) for probabilities below 0.20, Low-Risk Zone (Yellow) for 0.20 to below 0.60, Medium-Risk Zone (Orange) for 0.60 to below 0.90, and High-Risk Zone (Red) for 0.90 and above.

The notebook prints the zone counts and reports 2,173 employees in the Safe Zone, 122 in the Low-Risk Zone, 53 in the Medium-Risk Zone, and 652 in the High-Risk Zone within the evaluated test-result frame. This makes the project useful not only for model benchmarking but also for downstream HR decision support.

Python Learning Inside This Project
This project contains multiple Python learning layers beyond the final model output. It is especially useful as a learning artifact because it shows how a real ML notebook is structured from import statements to business recommendations.

1. Python Libraries and Imports
The notebook uses several important Python libraries for data science and machine learning:

numpy for numeric handling

pandas for data loading, filtering, preprocessing, and dataframe operations

matplotlib.pyplot and seaborn for visual analysis

sklearn.cluster.KMeans for clustering

sklearn.model_selection tools such as train_test_split, StratifiedKFold, and cross_val_predict

sklearn.linear_model.LogisticRegression

sklearn.ensemble.RandomForestClassifier and GradientBoostingClassifier

sklearn.metrics tools such as classification report, confusion matrix, ROC curve, and ROC-AUC score

imblearn.over_sampling.SMOTE for imbalance correction

2. DataFrame Operations
The project teaches practical pandas operations such as:

Reading CSV files with pd.read_csv()

Inspecting head rows and schema

Selecting numeric and categorical columns using select_dtypes()

Dropping columns

Combining dataframes using pd.concat()

Creating result tables for predicted probabilities and risk labels

3. Preprocessing Skills
The project demonstrates several preprocessing concepts in Python:

Separating target and feature variables

Converting categorical variables into numeric features with pd.get_dummies()

Building a final model-ready feature matrix

Using stratified splitting to preserve target balance in train and test datasets

Applying SMOTE only to training data

4. Visualization Skills
The notebook includes Python-based EDA and model-understanding visualizations such as:

Correlation heatmaps

Distribution plots

Bar plots for comparing turnover across project counts

Scatter-based cluster visualization after K-Means

ROC curve analysis for model comparison

5. Unsupervised Learning Skills
The K-Means section teaches how to:

Filter a subgroup of interest

Select relevant clustering features

Fit a KMeans model

Retrieve cluster labels

Plot clusters visually and write business-level cluster interpretations

6. Supervised Learning Skills
The model-training section teaches:

Binary classification problem formulation

Baseline vs advanced-model comparison

How to use Logistic Regression, Random Forest, and Gradient Boosting in Python

How to train models on balanced data

How to evaluate models with cross-validation instead of relying on one split

7. Evaluation and Model Selection Skills
The project develops strong evaluation understanding through:

Classification reports with precision, recall, F1-score, and support

Confusion-matrix interpretation

ROC and ROC-AUC comparison

Choosing the best metric based on business cost, especially prioritizing recall because missed attrition cases are more harmful than extra alerts

8. Business Translation Skills
A particularly useful learning aspect is that the notebook does not stop at model accuracy. It converts output probabilities into HR-friendly risk zones, which is an example of turning machine learning predictions into actionable business categories.

This demonstrates an important professional skill: model results must often be translated into operational decision frameworks rather than only reported as metrics.

Main Project Workflow
The Python notebook follows this practical workflow:

Import libraries and load the HR dataset

Inspect the data and verify there are no missing values

Explore important variables using plots and correlation analysis

Cluster employees who left using K-Means

Convert categorical features into numeric form

Split the data with stratification

Apply SMOTE on the training data

Train Logistic Regression, Random Forest, and Gradient Boosting models using 5-fold CV

Compare models using classification report, ROC-AUC, and confusion matrix

Select the best model and create risk-zone-based retention outputs

Key Results
The major findings from the implemented notebook are:

No missing values were found in the dataset

Employees who left can be explored through satisfaction, evaluation, and project-load patterns

K-Means on employees who left forms three broad satisfaction-evaluation clusters

SMOTE successfully balances the minority attrition class in training data from 2,857 to 9,142

Random Forest is the best-performing model with AUC 0.994 and a very strong confusion matrix

Recall is treated as especially important because missing a true attrition case is more costly than raising an extra retention alert

The final output supports risk-zone-based employee retention planning

Why This Project Matters
This project is a strong portfolio example because it combines data cleaning, visualization, clustering, imbalance correction, supervised learning, cross-validation, metric-driven model selection, and business interpretation in one notebook. It shows both machine learning understanding and applied Python execution.

It is also valuable from a learning perspective because it demonstrates how a real-world ML problem is solved in stages: understand the problem, inspect the data, explore patterns, prepare features, handle imbalance, train several models, compare them fairly, and translate the results into business action.
