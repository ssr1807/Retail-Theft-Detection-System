"""Simple rule-based risk scoring for tracked retail customers."""


# ---------------------------- RISK CONSTANTS ---------------------------- #

# Keeping rule values in one place makes the logic easy to explain and tune.
SAME_SHELF_FRAME_LIMIT = 90
SAME_SHELF_RISK = 20

REPEATED_SHELF_VISIT_LIMIT = 3
REPEATED_SHELF_RISK = 15

SHELF_TO_EXIT_RISK = 25

HIGH_RISK_THRESHOLD = 60
HIGH_RISK_SHELF_FRAME_LIMIT = 30
HIGH_RISK_SHELF_RISK = 15

CHECKOUT_RISK_REDUCTION = 20
MIN_RISK = 0
MAX_RISK = 100

EXIT_ZONE = "EXIT"
CHECKOUT_ZONE = "CHECKOUT"
SHELF_ZONE_PREFIX = "SHELF_"


# Stores the behavior history of every customer, indexed by DeepSORT track ID.
person_history = {}


def _is_shelf(zone):
    """Return True for zones such as SHELF_A, SHELF_B and SHELF_C."""
    return zone.startswith(SHELF_ZONE_PREFIX)


def _new_customer_history():
    """Create an independent history dictionary for a newly seen customer."""
    return {
        "total_frames": 0,
        "current_zone": None,
        "previous_zone": None,
        "zone_history": [],
        "consecutive_frames_in_same_zone": 0,
        # A dictionary is used because visits must be counted per shelf.
        "shelf_visit_count": {},
        "exit_visit_count": 0,
        "total_risk": 0,
        "total_shelf_frames": 0,
        "last_event": "NORMAL",
    }


def update_risk(track_id, zone):
    """Update and return the behavior-based risk score for one customer.

    The function is called once per tracked video frame. Risk events based on
    entering a zone are therefore checked only when the zone changes. This
    prevents the same event from adding or subtracting risk on every frame.
    """
    if track_id not in person_history:
        person_history[track_id] = _new_customer_history()

    history = person_history[track_id]
    zone = str(zone).upper()

    # Save the old zone before recording the customer's current position.
    previous_zone = history["current_zone"]
    zone_changed = previous_zone != zone

    history["total_frames"] += 1
    history["previous_zone"] = previous_zone
    history["current_zone"] = zone
    history["zone_history"].append(zone)

    if zone_changed:
        history["consecutive_frames_in_same_zone"] = 1
    else:
        history["consecutive_frames_in_same_zone"] += 1

    current_zone_frames = history["consecutive_frames_in_same_zone"]
    current_risk = history["total_risk"]
    was_already_high_risk = current_risk >= HIGH_RISK_THRESHOLD
    if _is_shelf(zone):
        history["total_shelf_frames"] += 1

    entered_shelf = zone_changed and _is_shelf(zone)
    entered_exit = zone_changed and zone == EXIT_ZONE
    entered_checkout = zone_changed and zone == CHECKOUT_ZONE

    # Count each shelf entry once. The first observed shelf is the first visit;
    # later entries to that same shelf are return visits.
    shelf_visits = 0
    if entered_shelf:
        visits = history["shelf_visit_count"]
        visits[zone] = visits.get(zone, 0) + 1
        shelf_visits = visits[zone]

    if entered_exit:
        history["exit_visit_count"] += 1

    # Rule 1: add risk once when a continuous shelf stay passes 90 frames.
    if _is_shelf(zone) and current_zone_frames == SAME_SHELF_FRAME_LIMIT + 1:
        current_risk += SAME_SHELF_RISK

    # Rule 2: add risk for each new visit after the third visit to that shelf.
    if entered_shelf and shelf_visits > REPEATED_SHELF_VISIT_LIMIT:
        current_risk += REPEATED_SHELF_RISK

    # Rule 3: detect a direct transition from any shelf to the exit.
    if entered_exit and previous_zone is not None and _is_shelf(previous_zone):
        current_risk += SHELF_TO_EXIT_RISK

    # Rule 4: a customer who was already high-risk becomes more suspicious
    # after continuing at the same shelf for another meaningful period.
    if (
        was_already_high_risk
        and _is_shelf(zone)
        and current_zone_frames == HIGH_RISK_SHELF_FRAME_LIMIT
    ):
        current_risk += HIGH_RISK_SHELF_RISK

    # Rule 5: reaching checkout is a positive action. Apply the reduction once
    # on entry instead of repeatedly reducing risk on every checkout frame.
    if entered_checkout:
        current_risk -= CHECKOUT_RISK_REDUCTION

    # Rule 6: every returned score must remain in the valid 0-100 range.
    history["total_risk"] = max(MIN_RISK, min(current_risk, MAX_RISK))
    return history["total_risk"]