# Breast Cancer Diagnosis Classification - ML Assignment 2

## a. Problem Statement

Early, accurate diagnosis of breast tumours as benign or malignant is
critical for treatment decisions. This project builds and compares five
supervised classification models that predict a tumour's diagnosis from 30
numeric features computed from a digitized image of a fine needle aspirate
(FNA) of a breast mass, and exposes the trained models through an
interactive Streamlit app so predictions and evaluation metrics can be
explored without touching code.

## b. Dataset Description

- *Source:* Breast Cancer Data Set - originally
  from the UCI Machine Learning Repository, also widely mirrored on Kaggle
  as [Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).
- *File used:* `data/breast_cancer.csv` 
- *Instances:* 569 (exceeds the 500-instance minimum)
- *Features:* The dataset has 30 numerical features, so it satisfies the minimum requirement of 12 features. These features are based on 10 measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. For each measurement, three values are given: mean, standard error (SE), and worst value.
- *Target:* The target column is diagnosis. It contains two classes, M (malignant) and B (benign). So, the problem is a binary classification problem where the model has to predict whether a case is malignant or benign.
- *Class balance:* 357 benign (62.7%) vs. 212 malignant (37.3%) -
  Did not consider accuracy alone for comparing the models. I also looked at recall for the malignant class and MCC, since predicting a malignant case as benign is more serious.
- *Preprocessing:* all 30 features are continuous and on very different
  scales (e.g. `area_mean` is in the hundreds, `smoothness_mean` is around
  0.1), so they are standardized (`sklearn.preprocessing.StandardScaler`,
  fit on the training set only) before being fed to every model. The target
  is label-encoded (`B`→0, `M`→1).
- *Train/test split:* stratified 80/20 split, `random_state=42`. The held
  out 20% (110+ rows) is exported as `test_data.csv`.

## c. GitHub Repository Link

> `https://github.com/ranjan-corp-ghub-1526/ML`

- Screenshots uploaded in the Repository

## d. Streamlit Link
> https://ranjan2025ac05093.streamlit.app/

## e. Models Used

Five classifiers are trained on the identical train/test split in
`model/train_models.py`:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (k=5)
4. Naive Bayes Classifier (Gaussian - fits standardized continuous features
   naturally)
5. Random Forest Classifier (Ensemble, 200 trees)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9573 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9454 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9077 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest (Ensemble) | 0.9649 | 0.9524 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best all-round performer despite being the simplest model — highest recall (92.86%) and F1 (0.951) of all five, tied for highest accuracy (96.49%), and the highest AUC (0.957). Suggests the two classes are close to linearly separable once features are standardized. |
| Decision Tree | Weakest of the five on almost every metric (accuracy 92.98%, AUC 0.925, MCC 0.849) — a single tree overfits to specific splits and doesn't generalize as well as an ensemble of many trees. |
| kNN | Strong precision (97.44%, close behind Random Forest) but recall (90.48%) trails Logistic Regression, meaning it misses slightly more malignant cases. Performs well because distance-based methods benefit a lot from the standardized features. |
| Naive Bayes | Lowest score on every single metric (AUC 0.908, F1 0.889, MCC 0.829, recall 85.71%). Its core assumption — that features are independent given the class — doesn't hold well here, since many tumour measurements (radius, perimeter, area) are naturally correlated with each other. |
| Random Forest (Ensemble) | Highest precision of all models — a perfect 1.0, meaning zero benign cases were misclassified as malignant on this test set — and the highest MCC (0.926), reflecting the benefit of averaging 200 trees over one. Recall (90.48%) is a touch below Logistic Regression, so it's marginally more likely to miss a malignant case in exchange for never raising a false alarm. |
| *Overall Winner for your dataset?* | *Logistic Regression*, by a narrow margin over Random Forest. LR leads on recall, F1, and AUC; Random Forest leads on precision and MCC — genuinely close. In a cancer-diagnosis context, missing a malignant tumor (false negative) is usually costlier than a false alarm, so LR's higher recall tips it as the winner here, though Random Forest is a very reasonable alternative pick if precision/MCC matter more for your framing. |


## How to Run

1. *Install dependencies*
   ```bash
   pip install -r requirements.txt
   ```
2. *Train the models* 
   ```bash
   python model/train_models.py
   ```
3. *Test the app locally*
   ```bash
   python -m streamlit run app.py
   ```

