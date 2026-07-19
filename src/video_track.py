import argparse
import math
from evaluation import evaluate_video
from pathlib import Path
from typing import Optional, Sequence, Tuple
import evaluation
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from alert_logic import update_suspicion
from alerts import save_alert
from risk_engine import update_risk
from zones import get_zone


DEFAULT_VIDEO_PATH = Path(
    r"D:\Datasets\CARR\Videos\Alif Store\Turning to Shelf\Alif Store-Turning to Shelf-1.mp4"
)
DEFAULT_OUTPUT_PATH = Path(
    r"D:\Retail Theft Detection\outputs\tracking\carr_output.mp4"
)
DEFAULT_MODEL_PATH = "yolov8n.pt"
DEFAULT_FPS = 30.0
WINDOW_NAME = "Retail Tracking"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and track people in retail video footage."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output MP4 video.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="YOLOv8 model name or path.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening the preview window.",
    )
    return parser.parse_args(argv)


def get_valid_fps(capture: cv2.VideoCapture) -> float:
    """Return trustworthy source FPS or a playback-safe fallback."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        return DEFAULT_FPS
    return fps


def create_writer(
    output_path: Path,
    fps: float,
    frame_size: Tuple[int, int],
) -> cv2.VideoWriter:
    """Create an MP4 writer and fail early if OpenCV cannot open it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size, True)

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "Could not open the output video writer. "
            f"Check the output path and mp4v codec support: {output_path}"
        )

    return writer


def process_video(
    video_path: Path,
    output_path: Path,
    model_path: str,
    display: bool = True,
) -> None:
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("mp4v output must use an .mp4 file extension.")

    Path("reports").mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)
    evaluation.start_timer()
    capture = cv2.VideoCapture(str(video_path))
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    writer: Optional[cv2.VideoWriter] = None
    frames_written = 0

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video: {video_path}")

        fps = get_valid_fps(capture)
        print("Processing video...")

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.size == 0:
                continue
            evaluation.log_frame()
            frame_height, frame_width = frame.shape[:2]
            # ---------- DRAW STORE ZONES ---------- #

            cv2.line(
                frame,
                (int(frame_width * 0.35), 0),
                (int(frame_width * 0.35), frame_height),
                (255, 0, 0),
                2
            )

            cv2.line(
                frame,
                (int(frame_width * 0.70), 0),
                (int(frame_width * 0.70), frame_height),
                (255, 0, 0),
                2
            )

            cv2.line(
                frame,
                (int(frame_width * 0.85), 0),
                (int(frame_width * 0.85), frame_height),
                (0, 0, 255),
                2
            )

            cv2.putText(frame, "SHELF A", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, "SHELF B", (int(frame_width*0.40), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, "SHELF C", (int(frame_width*0.73), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(frame, "EXIT", (int(frame_width*0.88), 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "CHECKOUT", (20, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Initialize from a decoded frame, since container metadata can be wrong.
            if writer is None:
                writer = create_writer(
                    output_path,
                    fps,
                    (frame_width, frame_height),
                )
                print(f"Writer opened: {writer.isOpened()}")
                print(f"Output: {output_path.resolve()}")

            results = model(frame, classes=[0], verbose=False)
            detections = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                detections.append(
                    (
                        [
                            int(x1),
                            int(y1),
                            int(x2 - x1),
                            int(y2 - y1),
                        ],
                        confidence,
                        "person",
                    )
                )

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                evaluation.log_customer(track_id)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                evaluation.customer_positions.append((center_x, center_y))

                zone = get_zone(
                    center_x,
                    center_y,
                    frame_width,
                    frame_height
                )
                risk = update_risk(track_id, zone)
                evaluation.log_risk(risk)
                suspicion_count = update_suspicion(track_id, zone, risk)

                alert = ""
                if suspicion_count >= 5:
                    alert = "THEFT ALERT"
                    save_alert(track_id, risk)
                    evaluation.log_alert(track_id)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                if risk >= 60:
                    cv2.putText(
                        frame,
                        "SUSPICIOUS",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"Customer {track_id} | {zone} | Risk: {risk}% {alert}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            writer.write(frame)
            frames_written += 1

            if display:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if writer is None:
            raise RuntimeError(
                "The input opened, but OpenCV could not decode any video frames."
            )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    evaluation.end_timer()
    # ---------------- AUTOMATIC SOFTWARE EVALUATION ---------------- #

    video_name = str(video_path).lower()

    if "no interest" in video_name:
        ground_truth = "normal"

    elif "viewing" in video_name:
        ground_truth = "normal"

    elif "touching" in video_name:
        ground_truth = "normal"

    elif "turning to shelf" in video_name:
        ground_truth = "normal"

    elif "picking and returning" in video_name:
        ground_truth = "normal"

    elif "picking and putting" in video_name:
        ground_truth = "suspicious"

    else:
        ground_truth = "normal"


    metrics = evaluation.get_runtime_metrics()

    if metrics["maximum_risk"] >= 60:
        prediction = "suspicious"
    else:
        prediction = "normal"

    evaluation.evaluate_video(
        ground_truth,
        prediction
    )
    metrics = evaluation.save_metrics()

    evaluation.add_video_result(
        video_path.stem,
        ground_truth,
        prediction
    )
            

    print("\n========== SOFTWARE EVALUATION ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("=========================================\n")
    print(f"Tracking complete. Frames written: {frames_written}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    process_video(
        video_path=args.video.expanduser(),
        output_path=args.output.expanduser(),
        model_path=args.model,
        display=not args.no_display,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



