# Retail Theft Detection System

A real-time computer vision project developed as a B.Tech training project to detect potentially suspicious customer behaviour in retail environments using **YOLOv8**, **DeepSORT**, and **rule-based behavioural analysis**.

The system detects and tracks customers across video frames, estimates behavioural risk based on predefined rules, and generates alerts for suspicious activities.

---

## Features

- Real-time person detection using YOLOv8
- Multi-object tracking using DeepSORT
- Rule-based behavioural risk analysis
- Customer ID assignment and tracking
- Risk score calculation based on movement patterns
- Theft alert generation
- Performance evaluation using classification metrics
- Visual reports including confusion matrices, heatmaps, FPS graphs, and risk distribution

---

## Technology Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- DeepSORT
- NumPy
- Pandas
- Matplotlib

---

## Project Structure

```
Retail-Theft-Detection-System/
│
├── src/                 # Source code
├── reports/             # Evaluation reports and metrics
├── docs/                # Project documentation
├── requirements.txt     # Python dependencies
└── README.md
```

---

## System Workflow

1. Read video input.
2. Detect customers using YOLOv8.
3. Track customers using DeepSORT.
4. Monitor customer movement and shelf interaction.
5. Calculate behavioural risk using predefined rules.
6. Generate theft alerts when the risk threshold is exceeded.
7. Produce evaluation reports and performance metrics.

---

## Behaviour Analysis

The current implementation estimates suspicious behaviour using handcrafted rules such as:

- Prolonged presence near shelves
- Repeated movement between predefined zones
- Continuous shelf interaction
- Accumulated behavioural risk over time

When the calculated risk exceeds a predefined threshold, the system generates a theft alert.

---

## Evaluation

The system was evaluated on multiple retail activity scenarios, including:

- No Interest
- Viewing
- Touching
- Turning to Shelf
- Picking and Returning
- Picking and Putting

Performance was analysed using:

- Accuracy
- Precision
- Recall
- F1 Score
- Specificity
- Confusion Matrix
- Processing FPS
- Risk Distribution

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ssr1807/Retail-Theft-Detection-System.git
cd Retail-Theft-Detection-System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main detection script:

```bash
python src/video_track.py
```

> Ensure that the required video files and YOLO model weights are available before execution.

---

## Future Improvements

- Pose estimation for action recognition
- Multi-camera tracking
- Product interaction detection
- Person re-identification across cameras
- Deep learning based behaviour classification
- Real-time deployment on CCTV systems

---

## Disclaimer

This project was developed as part of a B.Tech training project for academic purposes. The behaviour analysis currently relies on predefined heuristic rules and is intended as a proof-of-concept rather than a production-ready retail surveillance solution.

---

## Author

Suryansh Pratap Singh

B.Tech Computer Science Engineering (Data Science)

Training Project – Retail Theft Detection System
