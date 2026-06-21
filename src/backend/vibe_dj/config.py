"""All tunables for the vibe_dj system."""

# --- Video ---
FPS_TARGET = 30
CAMERA_INDEX = 0

# --- Face / MediaPipe ---
FACE_MESH_MAX_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5
# Reference inter-pupillary distance (pixels) for scale normalization.
# Pitch amplitude is divided by (observed_ipd / REFERENCE_IPD) so that
# camera distance does not change the apparent bob amplitude.
REFERENCE_IPD_PX = 100.0

# --- Emotion ---
EMOTION_CADENCE = 3          # run DeepFace every Nth video frame
EMA_ALPHA = 0.35             # aggressive smoothing on the emotion vector

# --- Vibe DSP ---
WINDOW_LEN_S = 3.0           # rolling window for vibe analysis (seconds)
BOB_FREQ_LO = 0.5            # bandpass lower bound (Hz)
BOB_FREQ_HI = 3.0            # bandpass upper bound (Hz)
VIBE_PLV_WEIGHT = 0.6        # weight for phase-locking value
VIBE_PERIOD_WEIGHT = 0.4     # weight for period-match score
PERIOD_TOLERANCE = 0.20      # fractional tolerance for freq match

# --- Agent thresholds ---
VIBE_HIGH = 0.7
VIBE_LOW = 0.3
VALENCE_POSITIVE = 0.6
VALENCE_NEGATIVE = 0.3

# --- Playback ---
SAMPLE_RATE = 22050
TRACKS_DIR = "tracks"

# --- Main loop ---
FUSION_HZ = 10               # how often the main loop fuses signals (Hz)
