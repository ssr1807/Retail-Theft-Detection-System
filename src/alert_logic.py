person_history = {}

def update_suspicion(track_id, zone, risk):

    if track_id not in person_history:

        person_history[track_id] = 0

    if zone == "LEFT" and risk >= 60:

        person_history[track_id] += 1

    elif zone == "CENTER" and risk >= 70:

        person_history[track_id] += 1

    elif zone == "RIGHT" and risk >= 80:

        person_history[track_id] += 1

    else:

        person_history[track_id] = max(
            0,
            person_history[track_id]-1
        )

    return person_history[track_id]