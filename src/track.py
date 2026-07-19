from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from zones import get_zone
from risk_engine import update_risk
from alert_logic import update_suspicion
from alerts import save_alert

import cv2
import os


# ---------------- MODELS ---------------- #

model = YOLO("yolov8n.pt")

tracker = DeepSort(
    max_age=30
)


# ---------------- PATHS ---------------- #

input_folder = r"D:\Retail Theft Detection\dataset\MOT17\train\MOT17-02-FRCNN\img1"

output_folder = r"D:\Retail Theft Detection\outputs\tracking"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ---------------- FRAMES ---------------- #

frames = sorted(
    os.listdir(input_folder)
)[:100]


# ---------------- PROCESS ---------------- #

for frame_name in frames:

    frame_path = os.path.join(

        input_folder,

        frame_name

    )

    frame = cv2.imread(

        frame_path

    )

    if frame is None:

        continue


    # ---------- DETECTION ---------- #

    results = model(

        frame,

        classes=[0]

    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        confidence = float(

            box.conf[0]

        )

        detections.append(

            (

                [

                    int(x1),

                    int(y1),

                    int(x2 - x1),

                    int(y2 - y1)

                ],

                confidence,

                "person"

            )

        )


    # ---------- TRACKING ---------- #

    tracks = tracker.update_tracks(

        detections,

        frame=frame

    )


    # ---------- LOOP THROUGH TRACKS ---------- #

    for track in tracks:

        if not track.is_confirmed():

            continue


        track_id = track.track_id


        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(

            int,

            ltrb

        )


        center_x = int(

            (x1 + x2) / 2

        )


        # ---------- ZONE ---------- #

        zone = get_zone(

            center_x,

            frame.shape[1]

        )


        # ---------- RISK ---------- #

        risk = update_risk(

            track_id,

            zone

        )


        # ---------- SUSPICION ---------- #

        suspicion_count = update_suspicion(

            track_id,

            zone,

            risk

        )


        # ---------- ALERT ---------- #

        alert = ""

        if suspicion_count >= 5:

            alert = "THEFT ALERT"

            save_alert(

                track_id,

                risk

            )


        # ---------- DRAW BOX ---------- #

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2

        )


        # ---------- SUSPICIOUS TEXT ---------- #

        if risk >= 60:

            cv2.putText(

                frame,

                "SUSPICIOUS",

                (x1, y2 + 20),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 0, 255),

                2

            )


        # ---------- LABEL ---------- #

        cv2.putText(

            frame,

            f"ID {track_id} | {zone} | {risk}% | {alert}",

            (x1, y1 - 10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (0, 255, 0),

            2

        )


    # ---------- SAVE FRAME ---------- #

    output_path = os.path.join(

        output_folder,

        frame_name

    )

    cv2.imwrite(

        output_path,

        frame

    )


print("Tracking complete.")