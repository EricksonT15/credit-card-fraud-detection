"""
evaluation.py

Reusable model evaluation functions for the Credit Card Fraud
Detection capstone project. Used in notebooks/03_modelling.ipynb
and notebooks/04_evaluation.ipynb.

Designed around imbalanced binary classification — accuracy is
intentionally NOT the headline metric. AUC-ROC, F1, and
Precision-Recall AUC are prioritised instead.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Compute the core evaluation metrics for a fitted binary classifier,
    evaluated on the ORIGINAL imbalanced test set (never resampled).

    Parameters
    ----------
    model : fitted classifier
        Must implement .predict() and .predict_proba()
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    model_name : str
        Label for printing/reporting.

    Returns
    -------
    dict
        Dictionary of key metrics for this model.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "auc_roc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 50}")
    print(f"  AUC-ROC               : {metrics['auc_roc']:.4f}")
    print(f"  Precision-Recall AUC  : {metrics['pr_auc']:.4f}")
    print(f"  F1-Score (fraud)      : {metrics['f1']:.4f}")
    print(f"  Precision (fraud)     : {metrics['precision']:.4f}")
    print(f"  Recall (fraud)        : {metrics['recall']:.4f}")

    return metrics


def compare_models(results: list) -> pd.DataFrame:
    """
    Build a comparison table from multiple evaluate_model() outputs.

    Parameters
    ----------
    results : list of dict
        Each dict is the output of evaluate_model().

    Returns
    -------
    pd.DataFrame
        Sorted comparison table (best AUC-ROC first).
    """
    df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)
    df = df.reset_index(drop=True)
    df.index += 1
    return df.round(4)


def plot_roc_curves(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    models_dict : dict
        {model_name: fitted_model}
    X_test, y_test : test data
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    plt.figure(figsize=(8, 7))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison", fontweight="bold", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_pr_curves(models_dict: dict, X_test, y_test, save_path: str = None):
    """
    Plot Precision-Recall curves for multiple models on the same axes.
    More informative than ROC when the positive class is rare.

    Parameters
    ----------
    models_dict : dict
        {model_name: fitted_model}
    X_test, y_test : test data
    save_path : str, optional
    """
    plt.figure(figsize=(8, 7))

    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        plt.plot(recall, precision, linewidth=2,
                  label=f"{name} (PR-AUC={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison", fontweight="bold", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, model_name: str = "Model",
                            save_path: str = None):
    """
    Plot a confusion matrix with fraud-relevant labelling.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
    save_path : str, optional
    """
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Legitimate", "Fraud"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontweight="bold", fontsize=13)
    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n{classification_report(y_test, y_pred, target_names=labels)}")
