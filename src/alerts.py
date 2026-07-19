saved_ids = set()

def save_alert(track_id, risk):

    if track_id in saved_ids:

        return

    saved_ids.add(track_id)

    with open(

        "reports/alerts_log.txt",

        "a"

    ) as file:

        file.write(

            f"Track ID {track_id} -> Risk {risk}%\n"

        )