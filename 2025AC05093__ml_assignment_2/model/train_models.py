import json
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

RANDOM_STATE_VALUE = 42
BASE_DIR_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR_PATH / "datasource" / "breast_cancer.csv"
MODEL_DIR = BASE_DIR_PATH / "model"
TEST_DATA_PATH = BASE_DIR_PATH / "test_data.csv"

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Make sure data/breast_cancer.csv exists."
        )
    return pd.read_csv(DATA_PATH)

def split_input_data(df: pd.DataFrame):
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE_VALUE, stratify=y)

def models_info():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE_VALUE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE_VALUE),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE_VALUE
        ),
    }
    
def main():
    df = load_data()
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First 20 rows of the dataset:\n{df.head(20)}")
    
    X_train, X_test, y_train, y_test = split_input_data(df)
    lbl_encoder = LabelEncoder().fit(["B", "M"])
    y_train_encoded = lbl_encoder.transform(y_train)
    y_test_encoded = lbl_encoder.transform(y_test)
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    stnd_scaler = StandardScaler()
    X_train_scaled = stnd_scaler.fit_transform(X_train)
    X_test_scaled = stnd_scaler.transform(X_test)
    print("Data scaling completed.")
    
    print(f"Saving the trained models and preprocessing objects to {MODEL_DIR}")
    with open(MODEL_DIR / "standard_scaler.pkl", "wb") as f:
        pickle.dump(stnd_scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(lbl_encoder, f)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X_train.columns), f)
    print("Preprocessing objects saved successfully.")
    
    model_evaluation_results = {}
    for model_name, model in models_info().items():
        model.fit(X_train_scaled, y_train_encoded)
        y_pred = model.predict(X_test_scaled)
        
        print(f"Saving the trained model: {model_name} to {MODEL_DIR / MODEL_FILENAMES[model_name]}")
        with open(MODEL_DIR / MODEL_FILENAMES[model_name], "wb") as f:
            pickle.dump(model, f)
        print(f"Model {model_name} saved successfully.")
        
        print(f"Evaluating model: {model_name}")
        accuracy = accuracy_score(y_test_encoded, y_pred)
        f1 = f1_score(y_test_encoded, y_pred)
        mcc = matthews_corrcoef(y_test_encoded, y_pred)
        precision = precision_score(y_test_encoded, y_pred)
        recall = recall_score(y_test_encoded, y_pred)
        roc_auc = roc_auc_score(y_test_encoded, y_pred)
        
        model_evaluation_results[model_name] = {
            "Accuracy": accuracy,
            "AUC": roc_auc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "MCC": mcc
        }
        
        print(f"Model Evaluation Results for {model_name}:")
        for metric, value in model_evaluation_results[model_name].items():
            print(f"{metric}: {value:.4f}")
        print("-" * 30)
        
    print("All models trained, evaluated, and saved successfully.")
    evaluate_model_df = pd.DataFrame(model_evaluation_results).T
    evaluate_model_df.index.name = "Model Name"
    evaluate_model_df = evaluate_model_df.round(5)
    print("Saving model evaluation results to CSV.")
    evaluate_model_df.to_csv(MODEL_DIR / "model_evaluation_results.csv")
    print("Model evaluation results saved successfully.")
    print("Model Evaluation Results:")
    print(evaluate_model_df.to_string())
    
main()