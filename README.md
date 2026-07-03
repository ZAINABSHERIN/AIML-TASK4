 Task 4: Classification with Logistic Regression

AI & ML Internship — Elevate Labs

 Objective
Build a binary classifier using logistic regression.

 Dataset
Breast Cancer Wisconsin Diagnostic Dataset (`sklearn.datasets.load_breast_cancer`) — 569 samples, 30 features, binary target (malignant/benign).

 Tools
Python, Scikit-learn, Pandas, Matplotlib, Seaborn

 Approach
Train/test split (80/20, stratified) → standardized features → fit `LogisticRegression` → evaluated with confusion matrix, precision/recall, ROC-AUC → plotted sigmoid function → tuned decision threshold.

 Results
- **Accuracy:** 98%
- **ROC-AUC:** 0.995
- Precision/recall ~0.98–0.99 on both classes

 Files
- `task4_logistic_regression.ipynb` — code + outputs
- `confusion_matrix.png`, `roc_curve.png`, `sigmoid_curve.png`, `threshold_tuning.png`
