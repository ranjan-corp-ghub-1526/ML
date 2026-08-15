import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    with open(MODEL_DIR / "standard_scaler.pkl", "rb") as f:
        stnd_scaler = pickle.load(f)
    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        lbl_encoder = pickle.load(f)
    with open(MODEL_DIR / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    return models, stnd_scaler, lbl_encoder, feature_columns


def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis Classifier", layout="wide")
    st.title("Breast Cancer Diagnosis Classifier")
    st.caption(
        "ML Assignment 2 - Logistic Regression, Decision Tree, kNN, "
        "Naive Bayes and Random Forest on the Breast Cancer "
    )

    required_files = list(MODEL_FILES.values()) + [
        "standard_scaler.pkl",
        "label_encoder.pkl",
        "feature_columns.json",
    ]
    missing_files = [file for file in required_files if not (MODEL_DIR / file).exists()]

    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}")
        return

    models, standard_scaler, lbl_encoder, feature_columns = load_models()

    st.sidebar.header("Controls")
    model_name = st.sidebar.selectbox("Select a model", list(models.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload the test data (CSV)", type="csv")

    if uploaded_file is None:
        st.info("Upload a CSV to see predictions and metrics.")
        st.subheader("Expected columns")
        st.code(", ".join(feature_columns + ["diagnosis"]))
        return

    test_data = pd.read_csv(uploaded_file)
    missing_cols = [col for col in feature_columns if col not in test_data.columns]

    if missing_cols:
        st.error(f"Missing feature columns in the uploaded data: {', '.join(missing_cols)}")
        return

    desired_column = "diagnosis"
    if desired_column not in test_data.columns:
        st.error(f"Missing output column in the uploaded data: {desired_column}")
        return
    else:
        model = models[model_name]

        X_test = test_data[feature_columns]
        Y_test = test_data[desired_column]

        X_test_scaled = standard_scaler.transform(X_test)
        Y_prediction = model.predict(X_test_scaled)
        Y_prediction_labels = lbl_encoder.inverse_transform(Y_prediction)

        prediction_result = test_data.copy()
        prediction_result["Diagnosis (Predicted)"] = Y_prediction_labels
        st.subheader("Predictions (first 300 rows)")
        st.dataframe(prediction_result.head(300))

        Y_true = lbl_encoder.transform(Y_test)
        Y_probability = model.predict_proba(X_test_scaled)[:, 1]

        evaluation_metrics = {
            "Accuracy": accuracy_score(Y_true, Y_prediction),
            "AUC": roc_auc_score(Y_true, Y_probability),
            "Precision": precision_score(Y_true, Y_prediction),
            "Recall": recall_score(Y_true, Y_prediction),
            "F1": f1_score(Y_true, Y_prediction),
            "MCC": matthews_corrcoef(Y_true, Y_prediction),
        }
        st.subheader(f"Evaluation Metrics - {model_name}")
        cols = st.columns(len(evaluation_metrics))
        for col, (k, v) in zip(cols, evaluation_metrics.items()):
            col.metric(k, f"{v:.3f}")

        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(Y_true, Y_prediction)
        fig, ax = plt.subplots()

        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=lbl_encoder.classes_,
            yticklabels=lbl_encoder.classes_,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        class_report = classification_report(
            Y_true,
            Y_prediction,
            target_names=list(lbl_encoder.classes_),
            output_dict=True,
        )
        st.dataframe(pd.DataFrame(class_report).T.round(5))


main()
