"""Evaluation utilities for the Retail Theft Detection System.

The functions in this module can be called from ``video_track.py`` while a
video is being processed. Calling :func:`save_metrics` creates the text report,
CSV file, graphs, confusion matrix and position heat map.
"""

import csv
import math
import time
from pathlib import Path

import matplotlib

# Save images without opening GUI windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

overall_results = []
# ----------------------- CONFUSION MATRIX VALUES ----------------------- #

# Replace these values after comparing predictions with ground truth.
TP = 0
TN = 0
FP = 0
FN = 0
def evaluate_video(ground_truth, prediction):
    """
    Update TP, TN, FP and FN after processing one complete video.

    ground_truth:
        "normal"
        "suspicious"

    prediction:
        "normal"
        "suspicious"
    """

    global TP, TN, FP, FN

    ground_truth = ground_truth.lower()
    prediction = prediction.lower()

    if ground_truth == "normal" and prediction == "normal":
        TN += 1

    elif ground_truth == "normal" and prediction == "suspicious":
        FP += 1

    elif ground_truth == "suspicious" and prediction == "normal":
        FN += 1

    elif ground_truth == "suspicious" and prediction == "suspicious":
        TP += 1
        #--------------------- RUNTIME STATISTICS -------------------------- #

total_frames = 0
total_customers_detected = 0
unique_customer_ids = set()
alerts_generated = 0

risk_values = []
fps_values = []
customer_positions = []

_logged_alert_ids = set()
_start_time = None
_end_time = None
_last_frame_time = None


def _reset_runtime_data():
    """Clear measurements from an earlier video-processing run."""
    global total_frames, total_customers_detected, alerts_generated
    global _start_time, _end_time, _last_frame_time

    total_frames = 0
    total_customers_detected = 0
    alerts_generated = 0

    unique_customer_ids.clear()
    risk_values.clear()
    fps_values.clear()
    customer_positions.clear()
    _logged_alert_ids.clear()

    _start_time = None
    _end_time = None
    _last_frame_time = None


def start_timer():
    """Start a fresh evaluation run and return its start time."""
    global _start_time, _last_frame_time

    _reset_runtime_data()
    _start_time = time.perf_counter()
    _last_frame_time = _start_time
    return _start_time


def end_timer():
    """Stop timing and return the total processing time in seconds."""
    global _end_time

    if _start_time is None:
        return 0.0

    if _end_time is None:
        _end_time = time.perf_counter()

    return _end_time - _start_time


def _current_runtime():
    """Return elapsed time without unnecessarily stopping the timer."""
    if _start_time is None:
        return 0.0

    finish_time = _end_time if _end_time is not None else time.perf_counter()
    return finish_time - _start_time


def log_frame():
    """Record one processed frame and its instantaneous processing FPS."""
    global total_frames, _start_time, _last_frame_time

    current_time = time.perf_counter()

    # Automatically begin timing if start_timer() was omitted.
    if _start_time is None:
        _start_time = current_time
        _last_frame_time = current_time

    total_frames += 1

    if _last_frame_time is not None:
        frame_time = current_time - _last_frame_time
        if frame_time > 0:
            fps_values.append(1.0 / frame_time)

    _last_frame_time = current_time


def log_customer(track_id):
    """Record one customer observation and remember the unique track ID."""
    global total_customers_detected

    total_customers_detected += 1

    if track_id is not None:
        unique_customer_ids.add(track_id)


def log_alert(alert_id=None):
    """Record an alert.

    Passing a track ID prevents the same customer alert from being counted
    repeatedly on every frame.
    """
    global alerts_generated

    if alert_id is not None:
        if alert_id in _logged_alert_ids:
            return

        _logged_alert_ids.add(alert_id)

    alerts_generated += 1


def log_risk(risk):
    """Record one numeric risk score."""
    try:
        risk = float(risk)
    except (TypeError, ValueError):
        return

    if math.isfinite(risk):
        risk_values.append(risk)


def add_position(x, y):
    """Record a customer center coordinate for the heat map."""
    try:
        x = float(x)
        y = float(y)
    except (TypeError, ValueError):
        return

    if math.isfinite(x) and math.isfinite(y):
        customer_positions.append((x, y))


def set_confusion_values(tp, tn, fp, fn):
    """Set TP, TN, FP and FN after comparison with ground truth."""
    global TP, TN, FP, FN

    values = (tp, tn, fp, fn)

    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("TP, TN, FP and FN must be non-negative integers.")

    TP, TN, FP, FN = values


def _confusion_values_exist():
    """Check whether all confusion matrix values are available."""
    return all(value is not None for value in (TP, TN, FP, FN))


def _safe_divide(numerator, denominator):
    """Prevent division-by-zero errors."""
    return numerator / denominator if denominator else 0.0


def _classification_metrics():
    """Calculate classification metrics from TP, TN, FP and FN."""
    empty_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "specificity": None,
    }

    if not _confusion_values_exist():
        return empty_metrics

    # Reconstruct labels so sklearn.metrics can process the four counts.
    y_true = [1] * TP + [0] * TN + [0] * FP + [1] * FN
    y_pred = [1] * TP + [0] * TN + [1] * FP + [0] * FN

    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "recall": recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "f1_score": f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
            "specificity": recall_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            ),
        }

    except ImportError:
        # Formula fallback keeps the module working without scikit-learn.
        accuracy = _safe_divide(TP + TN, TP + TN + FP + FN)
        precision = _safe_divide(TP, TP + FP)
        recall = _safe_divide(TP, TP + FN)
        specificity = _safe_divide(TN, TN + FP)
        f1 = _safe_divide(
            2 * precision * recall,
            precision + recall,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
        }


def _collect_metrics():
    """Build the final metrics dictionary."""
    runtime = _current_runtime()
    average_fps = _safe_divide(total_frames, runtime)
    average_risk = _safe_divide(
        sum(risk_values),
        len(risk_values),
    )
    maximum_risk = max(risk_values, default=0.0)

    metrics = {
        "frames_processed": total_frames,
        "customers_detected": total_customers_detected,
        "unique_customers": len(unique_customer_ids),
        "average_fps": average_fps,
        "processing_time_seconds": runtime,
        "average_risk": average_risk,
        "maximum_risk": maximum_risk,
        "alerts_generated": alerts_generated,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
    }

    metrics.update(_classification_metrics())
    return metrics


def _format_optional_metric(value):
    """Format an optional classification value for the report."""
    if value is None:
        return "Not available"

    return f"{value:.4f}"


def _save_text_report(output_path, metrics):
    """Write evaluation_report.txt."""
    separator = "---------------------------------"

    lines = [
        separator,
        "Retail Theft Detection Evaluation",
        "",
        f"Frames Processed: {metrics['frames_processed']}",
        f"Customers Detected: {metrics['customers_detected']}",
        f"Unique Customer IDs: {metrics['unique_customers']}",
        f"Average FPS: {metrics['average_fps']:.2f}",
        (
            "Processing Time: "
            f"{metrics['processing_time_seconds']:.2f} seconds"
        ),
        f"Average Risk: {metrics['average_risk']:.2f}",
        f"Maximum Risk: {metrics['maximum_risk']:.2f}",
        f"Alerts Generated: {metrics['alerts_generated']}",
        "",
        (
            f"TP: {metrics['tp']}"
            if metrics["tp"] is not None
            else "TP: Not available"
        ),
        (
            f"TN: {metrics['tn']}"
            if metrics["tn"] is not None
            else "TN: Not available"
        ),
        (
            f"FP: {metrics['fp']}"
            if metrics["fp"] is not None
            else "FP: Not available"
        ),
        (
            f"FN: {metrics['fn']}"
            if metrics["fn"] is not None
            else "FN: Not available"
        ),
        f"Accuracy: {_format_optional_metric(metrics['accuracy'])}",
        f"Precision: {_format_optional_metric(metrics['precision'])}",
        f"Recall: {_format_optional_metric(metrics['recall'])}",
        f"F1 Score: {_format_optional_metric(metrics['f1_score'])}",
        f"Specificity: {_format_optional_metric(metrics['specificity'])}",
        separator,
    ]

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _save_csv(output_path, metrics):
    """Write all collected values to metrics.csv."""
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(metrics.keys()),
        )
        writer.writeheader()
        writer.writerow(metrics)


def _save_metrics_bar(output_path, metrics):
    """Create a bar graph of the main metrics."""
    labels = [
        "Frames",
        "Unique IDs",
        "Avg FPS",
        "Avg Risk",
        "Max Risk",
        "Alerts",
    ]

    values = [
        metrics["frames_processed"],
        metrics["unique_customers"],
        metrics["average_fps"],
        metrics["average_risk"],
        metrics["maximum_risk"],
        metrics["alerts_generated"],
    ]

    colors = [
        "#4472C4",
        "#70AD47",
        "#5B9BD5",
        "#FFC000",
        "#ED7D31",
        "#C00000",
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    bars = axis.bar(labels, values, color=colors)

    axis.set_title("Retail Theft Detection Metrics")
    axis.set_ylabel("Value")
    axis.bar_label(bars, fmt="%.2f", padding=3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_risk_distribution(output_path):
    """Create a histogram of recorded risk values."""
    figure, axis = plt.subplots(figsize=(8, 5))

    if risk_values:
        axis.hist(
            risk_values,
            bins=10,
            range=(0, 100),
            color="#ED7D31",
            edgecolor="black",
        )
        axis.set_ylabel("Number of Observations")
    else:
        axis.text(
            0.5,
            0.5,
            "No risk values recorded",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("Risk Distribution")
    axis.set_xlabel("Risk Score")
    axis.set_xlim(0, 100)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_fps_graph(output_path):
    """Create a graph of frame-by-frame processing FPS."""
    figure, axis = plt.subplots(figsize=(9, 5))

    if fps_values:
        axis.plot(
            range(1, len(fps_values) + 1),
            fps_values,
            color="#4472C4",
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No FPS values recorded",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("Processing FPS")
    axis.set_xlabel("Processed Frame")
    axis.set_ylabel("FPS")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_confusion_matrix(output_path):
    """Create the confusion matrix image."""
    figure, axis = plt.subplots(figsize=(6, 5))

    if _confusion_values_exist():
        matrix = [
            [TN, FP],
            [FN, TP],
        ]

        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)

        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    matrix[row][column],
                    ha="center",
                    va="center",
                    color="black",
                )

        axis.set_xticks(
            [0, 1],
            labels=["Normal", "Theft"],
        )
        axis.set_yticks(
            [0, 1],
            labels=["Normal", "Theft"],
        )
        axis.set_xlabel("Predicted Class")
        axis.set_ylabel("Actual Class")
    else:
        axis.text(
            0.5,
            0.5,
            "TP, TN, FP and FN not provided",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])

    axis.set_title("Confusion Matrix")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_heatmap(output_path):
    """Create a heat map from customer center coordinates."""
    figure, axis = plt.subplots(figsize=(9, 6))

    if customer_positions:
        x_values = [
            position[0]
            for position in customer_positions
        ]
        y_values = [
            position[1]
            for position in customer_positions
        ]

        heatmap = axis.hist2d(
            x_values,
            y_values,
            bins=30,
            cmap="hot",
        )

        figure.colorbar(
            heatmap[3],
            ax=axis,
            label="Customer Presence",
        )

        # Video coordinates start from the top-left corner.
        axis.invert_yaxis()
    else:
        axis.text(
            0.5,
            0.5,
            "No customer positions recorded",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    axis.set_title("Customer Position Heat Map")
    axis.set_xlabel("X Coordinate")
    axis.set_ylabel("Y Coordinate")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metrics(output_directory="reports"):
    """Save all reports and graphs.

    Args:
        output_directory: Folder in which the evaluation files are created.

    Returns:
        Dictionary containing the final calculated metrics.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Stop timing so every output uses the same final runtime.
    end_timer()
    metrics = _collect_metrics()

    _save_text_report(
        output_directory / "evaluation_report.txt",
        metrics,
    )
    _save_csv(
        output_directory / "metrics.csv",
        metrics,
    )
    _save_metrics_bar(
        output_directory / "metrics_bar.png",
        metrics,
    )
    _save_risk_distribution(
        output_directory / "risk_distribution.png",
    )
    _save_fps_graph(
        output_directory / "fps_graph.png",
    )
    _save_confusion_matrix(
        output_directory / "confusion_matrix.png",
    )
    _save_heatmap(
        output_directory / "heatmap.png",
    )

    return metrics
def get_runtime_metrics():
    """
    Return the current runtime statistics collected while processing a video.
    """

    processing_time = 0

    if _start_time is not None and _end_time is not None:
        processing_time = _end_time - _start_time

    average_fps = 0
    if processing_time > 0:
        average_fps = total_frames / processing_time

    average_risk = 0
    if risk_values:
        average_risk = sum(risk_values) / len(risk_values)

    maximum_risk = 0
    if risk_values:
        maximum_risk = max(risk_values)

    return {
        "frames_processed": total_frames,
        "customers_detected": total_customers_detected,
        "unique_customers": len(unique_customer_ids),
        "average_fps": average_fps,
        "processing_time_seconds": processing_time,
        "average_risk": average_risk,
        "maximum_risk": maximum_risk,
        "alerts_generated": alerts_generated,
    }
def add_video_result(video_name, ground_truth, prediction):
    overall_results.append({
        "video": video_name,
        "ground_truth": ground_truth,
        "prediction": prediction
    })