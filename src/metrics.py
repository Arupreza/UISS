import os
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from math import sqrt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib.colors import LinearSegmentedColormap


# Very light greenyellow colormap
light_greeyellow = LinearSegmentedColormap.from_list(
    "light_greenyellow",
    ["#FFFFFF", "#F5FFB0", "#DAFF70"]   # white → light → medium-light greenyellow
)

def PlotConfusionMatrix(cf_matrix, class_labels):
    cf_matrix = np.array(cf_matrix)
    n = len(class_labels)

    # ---- Calculate row percentages ---- #
    percentages = np.zeros_like(cf_matrix, dtype=float)
    for i in range(n):
        row_sum = cf_matrix[i].sum()
        if row_sum > 0:
            percentages[i] = (cf_matrix[i] / row_sum) * 100

    # ---- Normalize for heat intensity ---- #
    intensity_matrix = percentages / 100.0

    # ---- Plot heatmap ---- #
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        intensity_matrix,
        cmap=light_greeyellow,
        cbar=False,
        annot=False,
        linewidths=1.0,
        linecolor="black"
    )

    # ---- Overlay numbers + percentages ---- #
    for i in range(n):
        for j in range(n):
            ax.text(
                j + 0.5, i + 0.33,
                f"{cf_matrix[i][j]}",
                ha="center", va="center",
                fontsize=18, fontweight="bold", color="black"
            )
            ax.text(
                j + 0.5, i + 0.74,
                f"{percentages[i][j]:.2f}%",
                ha="center", va="center",
                fontsize=15, fontweight="bold", color="black"
            )

    # ---- Tick colors (updated to match renamed classes) ---- #
    tick_colors = {
        "S&AF": "green",
        "AB": "#FFA500", "AL": "#FFA500", "AA": "#FFA500",
        "DoS": "#800000", "Fuzz": "#800000", "Replay": "#800000",
        "AAB": "red", "AAL": "red", "AAA": "red"
    }

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)

    ax.set_xticklabels(
        class_labels, rotation=45, ha='right',
        fontsize=18, fontweight="bold"
    )
    ax.set_yticklabels(
        class_labels, rotation=0,
        fontsize=18, fontweight="bold"
    )

    # Apply tick colors
    for idx, lbl in enumerate(class_labels):
        ax.get_xticklabels()[idx].set_color(tick_colors[lbl])
        ax.get_yticklabels()[idx].set_color(tick_colors[lbl])

    # ---- Axis labels ---- #
    plt.xlabel("True Labels", fontsize=22, fontweight="bold", color="darkblue")
    plt.ylabel("Predicted Labels", fontsize=22, fontweight="bold", color="darkblue")

    plt.tight_layout()
    plt.show()

def ComputeFullMetrics(cf_matrix, class_labels, confidence=0.95):
    """
    Computes:
    - Precision
    - Recall
    - F1-score
    - Accuracy
    - Error rate
    - 95% CI
    - AUC per class (One-vs-Rest)

    Works purely from a confusion matrix.
    """
    cf = np.array(cf_matrix)
    n_classes = len(class_labels)

    # Prepare result table
    results = {
        "Class": [],
        "Support": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Accuracy": [],
        "Error_Rate": [],
        "CI_Lower": [],
        "CI_Upper": [],
        "AUC": []
    }

    # Z-score for confidence interval
    z = norm.ppf(1 - (1 - confidence) / 2)

    # ---- Compute Per-Class Metrics ---- #
    for i in range(n_classes):

        TP = cf[i, i]
        FN = np.sum(cf[i, :]) - TP
        FP = np.sum(cf[:, i]) - TP
        TN = np.sum(cf) - (TP + FP + FN)

        support = TP + FN

        # Precision, recall, F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Accuracy & error rate
        accuracy = TP / support if support > 0 else 0
        error_rate = 1 - accuracy

        # ---- Wilson Confidence Interval ---- #
        if support > 0:
            phat = accuracy
            denom = 1 + z*z/support
            centre = phat + z*z/(2*support)
            margin = z * sqrt((phat*(1-phat) + z*z/(4*support)) / support)

            CI_lower = (centre - margin) / denom
            CI_upper = (centre + margin) / denom
        else:
            CI_lower = CI_upper = 0

        # ---- AUC (One-vs-Rest) ---- #
        # Build binary GT and prediction scores
        y_true = np.repeat(np.arange(n_classes), cf.sum(axis=1))
        y_score = np.zeros_like(y_true, dtype=float)

        # Score = proportion of predicted class for this row
        counter = 0
        for r in range(n_classes):
            for _ in range(cf[r].sum()):
                y_score[counter] = cf[r][i] / cf[r].sum() if cf[r].sum() > 0 else 0
                counter += 1

        y_bin = (y_true == i).astype(int)

        try:
            auc = roc_auc_score(y_bin, y_score)
        except:
            auc = np.nan

        # ---- Store ---- #
        results["Class"].append(class_labels[i])
        results["Support"].append(int(support))
        results["Precision"].append(round(precision, 4))
        results["Recall"].append(round(recall, 4))
        results["F1"].append(round(f1, 4))
        results["Accuracy"].append(round(accuracy, 4))
        results["Error_Rate"].append(round(error_rate, 4))
        results["CI_Lower"].append(round(CI_lower, 4))
        results["CI_Upper"].append(round(CI_upper, 4))
        results["AUC"].append(round(auc, 4) if auc == auc else None)

    return pd.DataFrame(results)



def PlotPRCurve(
    y_true,
    y_score,
    class_labels,
    figsize=(8, 5),
    dpi=150,
    legend_loc="lower left",
    legend_ncol=2,
    legend_fontsize=12,   # <-- bigger legend text
    legend_alpha=0.25,
    xlim=(0.90, 1.00),
    ylim=(0.90, 1.00),
    xlabel="Recall",
    ylabel="Precision",
    axis_label_fontsize=16,
    axis_label_fontweight="bold",
    axis_label_color="darkblue",
):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    n_classes = len(class_labels)
    if y_score.ndim != 2 or y_score.shape[1] != n_classes:
        raise ValueError(f"y_score must have shape (N, {n_classes}). Got {y_score.shape}.")
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"y_true and y_score must have same N. Got {y_true.shape[0]} vs {y_score.shape[0]}.")

    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # Micro-average
    prec_micro, rec_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true_bin, y_score, average="micro")
    ax.plot(rec_micro, prec_micro, linewidth=2, label=f"micro-average (AP={ap_micro:.4f})")

    # Per-class
    ap_per_class = {}
    for c, name in enumerate(class_labels):
        prec_c, rec_c, _ = precision_recall_curve(y_true_bin[:, c], y_score[:, c])
        ap_c = average_precision_score(y_true_bin[:, c], y_score[:, c])
        ap_per_class[name] = float(ap_c)
        ax.plot(rec_c, prec_c, linewidth=1.5, label=f"{name} (AP={ap_c:.4f})")

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize, fontweight=axis_label_fontweight, color=axis_label_color)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize, fontweight=axis_label_fontweight, color=axis_label_color)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True)

    leg = ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        fontsize=legend_fontsize,  # <-- applies here
        frameon=True,
        fancybox=True
    )
    leg.get_frame().set_alpha(legend_alpha)
    leg.get_frame().set_edgecolor("gray")

    plt.tight_layout()
    plt.show()

    ap_macro = float(np.mean(list(ap_per_class.values()))) if ap_per_class else float("nan")
    return {"AP_micro": float(ap_micro), "AP_macro": ap_macro, "AP_per_class": ap_per_class}