FEATURE_NAMES: list[str] = [
    "zcr_mean", "zcr_std", "zcr_min", "zcr_max",
    "melspect_mean", "melspect_std", "melspect_min", "melspect_max",
    "mfcc_mean", "mfcc_std", "mfcc_min", "mfcc_max",
    "mfcc_d_mean", "mfcc_d_std", "mfcc_d_min", "mfcc_d_max",
    "mfcc_d2_mean", "mfcc_d2_std", "mfcc_d2_min", "mfcc_d2_max",
    "flux_mean", "flux_std", "flux_min", "flux_max",
    "flatness_mean", "flatness_std", "flatness_min", "flatness_max",
    "centroid_mean", "centroid_std", "centroid_min", "centroid_max",
    "bandwidth_mean", "bandwidth_std", "bandwidth_min", "bandwidth_max",
    "contrast_mean", "contrast_std", "contrast_min", "contrast_max",
    "rolloff_low_mean", "rolloff_low_std", "rolloff_low_min", "rolloff_low_max",
    "rolloff_high_mean", "rolloff_high_std", "rolloff_high_min", "rolloff_high_max",
    "energy_mean", "energy_std", "energy_min", "energy_max",
    "power_mean", "power_std", "power_min", "power_max",
]

METADATA_NAMES: list[str] = [
    "start_time",
    "end_time",
    "annotations",
    "is_own_recording",
    "class_names",
    "annotator_ids",
    "target_classes",
    "non_target_classes",
    "recording_device",
    "recording_environments",
    "scene_description",
    "device_placement",
]

CLASSES_NAMES: list[str] = [
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
import os as _os

ROOT: str = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATASET_DIR: str = _os.path.join(ROOT, "MLPC2026_dataset_development")
CACHE_PATH: str = _os.path.join(ROOT, "cache", "frames.npz")
RESULTS_DIR: str = _os.path.join(ROOT, "results")
FIGURES_DIR: str = _os.path.join(ROOT, "docs", "figures")
MODEL_DIR: str = _os.path.join(ROOT, "models")
EXAMPLES_DIR: str = _os.path.join(ROOT, "examples")

# ── Experiment protocol ───────────────────────────────────────────────────────
SEED: int = 42
# Fractions of *collectors* (people) assigned to each split. Every recording
# from a given collector lands in the same split, so test performance reflects
# unseen people, rooms and devices.
TRAIN_FRACTION: float = 0.70
VAL_FRACTION: float = 0.15
# An annotator's score must exceed this to count as a positive vote; a frame is
# labelled positive when strictly more than half of its annotators vote positive.
ANNOTATION_THRESHOLD: float = 0.25
