import logging
from pathlib import Path

import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

from backend.core.config import settings


def setup_logging():
    settings.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(settings.LOG_PATH),
        ],
    )


def load_data():
    df = pd.read_csv(settings.DATASET_PATH)
    logging.info(f"Dataset loaded — shape: {df.shape}")

    X = df.drop(columns=[settings.TARGET_COL])
    y = df[settings.TARGET_COL]

    return X, y


def get_train_test_split(X, y):
    row_signature = pd.util.hash_pandas_object(X, index=False)

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )
    train_idx, test_idx = next(gss.split(X, y, groups=row_signature))

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    logging.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_candidates(X_train, y_train):
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=1119,
            max_depth=5,
            max_features="sqrt",
            max_samples=0.6,
            min_samples_leaf=11,
            min_samples_split=30,
            ccp_alpha=0.0017,
            bootstrap=True,
            n_jobs=-1,
            random_state=settings.RANDOM_STATE,
        )),
    ])

    xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=settings.RANDOM_STATE,
            verbosity=0,
        )),
    ])

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=settings.RANDOM_STATE,
        )),
    ])

    voting = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("lr", lr),
        ],
        voting="soft",
    )

    return {
        "RandomForest": rf,
        "XGBoost": xgb,
        "LogisticRegression": lr,
        "VotingEnsemble": voting,
    }


def run_cross_validation(candidates, X_train, y_train):
    logging.info("Running 5-fold cross validation on all models...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=settings.RANDOM_STATE,
    )

    scoring = ["accuracy", "recall", "f1", "roc_auc"]
    results = {}

    for name, pipeline in candidates.items():
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        results[name] = {
            "accuracy": cv_results["test_accuracy"].mean(),
            "recall":   cv_results["test_recall"].mean(),
            "f1":       cv_results["test_f1"].mean(),
            "roc_auc":  cv_results["test_roc_auc"].mean(),
        }
        logging.info(
            f"{name} CV — "
            f"Accuracy: {results[name]['accuracy']:.4f} | "
            f"Recall: {results[name]['recall']:.4f} | "
            f"F1: {results[name]['f1']:.4f} | "
            f"ROC-AUC: {results[name]['roc_auc']:.4f}"
        )

    return results


def select_best_model(cv_results, candidates):
    best_name = max(cv_results, key=lambda k: cv_results[k]["roc_auc"])
    logging.info(f"Best model selected: {best_name}")
    return best_name, candidates[best_name]


def evaluate_on_test(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall":   recall_score(y_test, y_pred),
        "f1":       f1_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
    }

    logging.info(f"Test Accuracy : {metrics['accuracy']:.4f}")
    logging.info(f"Test Recall   : {metrics['recall']:.4f}")
    logging.info(f"Test F1       : {metrics['f1']:.4f}")
    logging.info(f"Test ROC-AUC  : {metrics['roc_auc']:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    return pipeline, metrics


def generate_shap(pipeline, X_train, best_name):
    if best_name == "VotingEnsemble":
        logging.info("SHAP skipped — VotingEnsemble not supported by TreeExplainer")
        return None

    logging.info("Generating SHAP values...")

    model  = pipeline.named_steps["model"]
    scaler = pipeline.named_steps["scaler"]
    X_scaled = scaler.transform(X_train)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    plt.figure()
    shap.summary_plot(
        shap_values if isinstance(shap_values, list) else shap_values,
        X_scaled,
        feature_names=X_train.columns.tolist(),
        show=False,
        plot_type="bar",
    )

    shap_path = settings.PROJECT_ROOT / settings.MODEL_DIR / "shap_summary.png"
    plt.savefig(shap_path, bbox_inches="tight", dpi=150)
    plt.close()
    logging.info(f"SHAP saved to {shap_path}")

    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def save_model(pipeline):
    settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, settings.MODEL_PATH)
    logging.info(f"Model saved to {settings.MODEL_PATH}")


def train_model():
    setup_logging()
    logging.info("=== MedBuddy.ML Training Pipeline Started ===")

    X, y                             = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    candidates                       = build_candidates(X_train, y_train)
    cv_results                       = run_cross_validation(candidates, X_train, y_train)
    best_name, best                  = select_best_model(cv_results, candidates)
    best, metrics                    = evaluate_on_test(best, X_train, y_train, X_test, y_test)

    generate_shap(best, X_train, best_name)
    save_model(best)

    logging.info("=== Training Pipeline Completed ===")
    return best_name, metrics


if __name__ == "__main__":
    train_model()