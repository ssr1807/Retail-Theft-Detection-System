person_history = {}

SUSPICIOUS_FRAME_LIMIT = 150

MOVEMENT_THRESHOLD = 40


def update_person(track_id, center_x, center_y):

    if track_id not in person_history:

        person_history[track_id] = {

            "frames": 0,

            "initial_x": center_x,

            "initial_y": center_y,

            "suspicious": False

        }

    person_history[track_id]["frames"] += 1

    dx = abs(

        center_x

        - person_history[track_id]["initial_x"]

    )

    dy = abs(

        center_y

        - person_history[track_id]["initial_y"]

    )

    movement = dx + dy

    if (

        person_history[track_id]["frames"]

        > SUSPICIOUS_FRAME_LIMIT

        and movement

        < MOVEMENT_THRESHOLD

    ):

        person_history[track_id]["suspicious"] = True

    return person_history[track_id]["suspicious"]