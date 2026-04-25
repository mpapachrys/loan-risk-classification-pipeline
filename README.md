#  Credit Risk Analysis: Predicting Loan Defaults with XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![F1 Score](https://img.shields.io/badge/F1--Score-0.8426-green)
![Recall](https://img.shields.io/badge/Recall-0.93-brightgreen)

##  Project Overview
This project focuses on building a high-precision machine learning pipeline to predict the likelihood of loan default. In the banking sector, identifying risky borrowers while maintaining a positive experience for creditworthy customers is a multi-million dollar challenge. 

Through rigorous data cleaning, outlier detection, and hyperparameter tuning, I developed a "Champion" model that achieves a **93% Recall rate**, ensuring that the vast majority of potential defaults are identified.

##  Key Results
| Metric | Score |
| :--- | :--- |
| **F1-Score (Class 1)** | **0.8426** |
| **Recall (Class 1)** | **93%** |
| **Precision (Class 1)** | **88%** |
| **Overall Accuracy** | **93%** |

##  Technical Journey & Challenges
One of the key findings in this project was the relationship between data dimensionality and model performance.

### The PCA/SVD Experiment
I initially attempted to use **Principal Component Analysis (PCA)** and **Singular Value Decomposition (SVD)** to reduce noise. However, this led to a performance crash (F1-score ~0.30).
* **The Reason:** Credit risk data relies on "sharp" binary indicators (e.g., `previous_loan_defaults`). Dimensionality reduction "smeared" these critical signals, turning clear red flags into gray noise.
* **The Solution:** I reverted to the **Full Feature Set**, allowing XGBoost's internal regularization to handle feature importance without losing the integrity of high-impact variables.

### Data Engineering Pipeline
1.  **Cleaning:** Handled missing values and synchronized the target labels with the cleaned feature set.
2.  **Outlier Removal:** Utilized **Isolation Forest** to prune anomalous data points that would otherwise skew the model's logic.
3.  **Preprocessing:** Implemented `PowerTransformer` for skewed numerical distributions and used a `ColumnTransformer` pipeline for seamless data flow.

##  Final Model Configuration
The winning model utilized **XGBoost** with the following optimized parameters:
* `max_depth=5`: Deep enough to capture complex interactions between income, age, and loan intent.
* `learning_rate=0.5`: Aggressive enough to map the sharp boundaries of credit risk.
* `scale_pos_weight=3.5`: Specifically tuned to handle the class imbalance (approx. 4:1 ratio of non-defaults to defaults).

##  Business Impact
This model successfully balances the bank's two main objectives:
1.  **Risk Mitigation:** Identifying 93% of potential defaulters before they cause financial loss.
2.  **Efficiency:** Maintaining an 88% precision rate to minimize "False Alarms," ensuring that reliable customers are not unfairly rejected for loans.

***

`./results.png`
