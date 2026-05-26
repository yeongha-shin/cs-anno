"""
Label codebook (single source of truth).

Both the auto-labeling pipeline and the review UI read from here.
Edit options/labels in one place and both sides update.

Each dimension:
  - key      : internal id
  - title    : display name (UI)
  - hotkeys  : keyboard shortcut -> option value (used in review UI)
  - options  : list of (value, label, definition)
               value is the machine value stored in JSON.

Annotator agreement comes from clear, observable definitions. The definition
text is shown to the reviewer as the criterion, so write it carefully.
"""

SCHEMA = {
    "attention": {
        "title": "Attention",
        "hotkeys": {"1": "1_low", "2": "2_medium", "3": "3_high"},
        "options": [
            ("1_low",    "1 Low",    "Gaze off screen/task, distracted, looking around, eyes closed"),
            ("2_medium", "2 Medium", "Looking at the task but passive; intermittently distracted"),
            ("3_high",   "3 High",   "Actively watching the screen or absorbed in the task"),
        ],
    },
    "emotion": {
        "title": "Emotion",
        "hotkeys": {"q": "happy", "w": "boring", "e": "frustrated", "r": "thinking"},
        "options": [
            ("happy",      "Happy",      "Lips form a smile / laughing"),
            ("boring",     "Boring",     "Eyes nearly closed (drowsy / slumping)"),
            ("frustrated", "Frustrated", "Pressed lips, scowl, lowered brows, negative tension"),
            ("thinking",   "Thinking",   "Eyes open, no smile or frown (neutral, default)"),
        ],
    },
    "action": {
        "title": "Action",
        "hotkeys": {"a": "watching", "s": "touching", "d": "speaking"},
        "options": [
            ("watching", "Watching", "Looking at the screen, hands not engaging the task (default)"),
            ("touching", "Touching", "Hand reaches the lower screen area (the tablet)"),
            ("speaking", "Speaking", "Mouth moves during the segment (talking)"),
        ],
    },
    "posture": {
        "title": "Posture",
        "hotkeys": {"z": "upright", "x": "chin_rest", "c": "face_touching"},
        "options": [
            ("upright",       "Upright",       "Head up, facing front / the screen (default)"),
            ("chin_rest",     "Chin rest",     "Hand supports the chin/cheek from below"),
            ("face_touching", "Face touching", "Hand touches the face (eyes/nose/forehead), not a chin support"),
        ],
    },
}

# Dimension order (UI display order)
DIMENSIONS = ["attention", "emotion", "action", "posture"]


def default_value(dim):
    """Neutral default for each dimension (fallback when auto-label is unsure)."""
    return {"attention": "2_medium", "emotion": "thinking",
            "action": "watching", "posture": "upright"}[dim]


def valid_values(dim):
    return [opt[0] for opt in SCHEMA[dim]["options"]]


def schema_json():
    """JSON-serializable form passed to the review UI (JS)."""
    return {
        "dimensions": DIMENSIONS,
        "dims": {
            dim: {
                "title": SCHEMA[dim]["title"],
                "hotkeys": SCHEMA[dim]["hotkeys"],
                "options": [
                    {"value": v, "label": label, "desc": desc}
                    for (v, label, desc) in SCHEMA[dim]["options"]
                ],
            }
            for dim in DIMENSIONS
        },
    }
