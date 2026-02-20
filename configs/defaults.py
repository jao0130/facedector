"""
YACS-based configuration system for unified face detection + rPPG project.
Reference: rPPG-Toolbox config pattern.

Usage:
    from configs.defaults import get_config
    cfg = get_config('configs/face_detection.yaml')
"""

from yacs.config import CfgNode as CN

_C = CN()

# ── Global ──
_C.TASK = "face_detection"  # "face_detection" | "rppg" | "realtime"
_C.DEVICE = "cuda:0"
_C.SEED = 42
_C.NUM_WORKERS = 8
_C.GPU_MEMORY_LIMIT_MB = 13000  # 80% of 16GB RTX 4060 Ti

# ── Face Detection Model ──
_C.FACE_MODEL = CN()
_C.FACE_MODEL.INPUT_SIZE = 256
_C.FACE_MODEL.BACKBONE = "mobilenetv2"
_C.FACE_MODEL.BACKBONE_ALPHA = 0.5
_C.FACE_MODEL.NUM_LANDMARKS = 5
_C.FACE_MODEL.FPN_CHANNELS = 64
_C.FACE_MODEL.PRETRAINED = True
_C.FACE_MODEL.WEIGHTS = ""  # path to .pth checkpoint

# ── rPPG Model (FCAtt) ──
_C.RPPG_MODEL = CN()
_C.RPPG_MODEL.NAME = "FCAtt"
_C.RPPG_MODEL.FRAMES = 128
_C.RPPG_MODEL.FPS = 30
_C.RPPG_MODEL.INPUT_SIZE = 72
_C.RPPG_MODEL.DROP_RATE1 = 0.1
_C.RPPG_MODEL.DROP_RATE2 = 0.2
_C.RPPG_MODEL.FREQ_ATT_POOL_SIZE = 18
_C.RPPG_MODEL.FREQ_ATT_TEMPORAL_KERNEL = 7
_C.RPPG_MODEL.FREQ_LOW = 0.7
_C.RPPG_MODEL.FREQ_HIGH = 2.5
_C.RPPG_MODEL.SPO2_MIN = 85.0
_C.RPPG_MODEL.SPO2_RANGE = 15.0
_C.RPPG_MODEL.WEIGHTS = ""

# ── Face Detection Data ──
_C.FACE_DATA = CN()
_C.FACE_DATA.CLEANED_LABELS_DIR = "D:/cleaned_labels"
_C.FACE_DATA.PURE_DIR = "D:/PURE"
_C.FACE_DATA.LABELS_DIR = "D:/PURE_labels"
_C.FACE_DATA.WIDER_DIR = ""
_C.FACE_DATA.WIDER_LABELS_DIR = ""
_C.FACE_DATA.CELEBA_IMG_DIR = ""
_C.FACE_DATA.CELEBA_LABELS_DIR = ""
_C.FACE_DATA.W300_DIR = ""
_C.FACE_DATA.W300_LABELS_DIR = ""
_C.FACE_DATA.TRAIN_SPLIT = 0.8

# ── rPPG Data ──
_C.RPPG_DATA = CN()
_C.RPPG_DATA.DATASET = "PURE"
_C.RPPG_DATA.DATA_PATH = ""
_C.RPPG_DATA.CACHED_PATH = ""
_C.RPPG_DATA.CACHED_PATHS = []          # Multi-source labeled cache dirs
_C.RPPG_DATA.FS = 30
_C.RPPG_DATA.CHUNK_LENGTH = 128
_C.RPPG_DATA.LABEL_TYPE = "DiffNormalized"
_C.RPPG_DATA.DATA_TYPE = "DiffNormalized"
_C.RPPG_DATA.CROP_FACE = True
_C.RPPG_DATA.LARGE_BOX_COEF = 1.5
_C.RPPG_DATA.RESIZE_W = 72
_C.RPPG_DATA.RESIZE_H = 72
_C.RPPG_DATA.DO_PREPROCESS = False
_C.RPPG_DATA.TEST_DATASET = ""  # Independent test dataset name (e.g. "UBFC")
_C.RPPG_DATA.TEST_DATA_PATH = ""  # Path to test dataset (e.g. "D:/UBFC/DATASET_2")
_C.RPPG_DATA.TEST_CACHED_PATH = ""  # Path to test preprocessed data
_C.RPPG_DATA.TRAIN_BEGIN = 0.0
_C.RPPG_DATA.TRAIN_END = 0.8
_C.RPPG_DATA.VALID_BEGIN = 0.7
_C.RPPG_DATA.VALID_END = 0.9
_C.RPPG_DATA.TEST_BEGIN = 0.0
_C.RPPG_DATA.TEST_END = 1.0

# ── Augmentation (Face Detection) ──
_C.AUGMENTATION = CN()
_C.AUGMENTATION.ROTATION_RANGE = 30.0
_C.AUGMENTATION.SCALE_MIN = 0.8
_C.AUGMENTATION.SCALE_MAX = 1.2
_C.AUGMENTATION.BRIGHTNESS = 0.2
_C.AUGMENTATION.CONTRAST = 0.2
_C.AUGMENTATION.SATURATION = 0.2
_C.AUGMENTATION.HUE = 0.1
_C.AUGMENTATION.BLUR_PROB = 0.3
_C.AUGMENTATION.BLUR_LIMIT = 7
_C.AUGMENTATION.NOISE_PROB = 0.2
_C.AUGMENTATION.HORIZONTAL_FLIP = True
_C.AUGMENTATION.PERSPECTIVE_PROB = 0.3

# ── Training (Face Detection) ──
_C.FACE_TRAIN = CN()
_C.FACE_TRAIN.BATCH_SIZE = 24
_C.FACE_TRAIN.EPOCHS = 100
_C.FACE_TRAIN.LR = 0.001
_C.FACE_TRAIN.MIN_LR = 0.00001
_C.FACE_TRAIN.WARMUP_EPOCHS = 5
_C.FACE_TRAIN.WEIGHT_DECAY = 0.001
_C.FACE_TRAIN.GRAD_CLIP = 5.0
_C.FACE_TRAIN.BBOX_WEIGHT = 1.0
_C.FACE_TRAIN.LANDMARK_WEIGHT = 1.5
_C.FACE_TRAIN.CONFIDENCE_WEIGHT = 0.1
_C.FACE_TRAIN.EARLY_STOPPING_PATIENCE = 20

# ── Training (rPPG) ──
_C.RPPG_TRAIN = CN()
_C.RPPG_TRAIN.BATCH_SIZE = 4
_C.RPPG_TRAIN.EPOCHS = 50
_C.RPPG_TRAIN.LR = 9e-3
_C.RPPG_TRAIN.PRETRAIN_RATIO = 0.6
_C.RPPG_TRAIN.GRAD_ACCUMULATION = 4
_C.RPPG_TRAIN.GRAD_CLIP = 1.0
_C.RPPG_TRAIN.PATIENCE_BVP = 30
_C.RPPG_TRAIN.PATIENCE_SPO2 = 20
_C.RPPG_TRAIN.USE_LAST_EPOCH = False

# ── Semi-supervised rPPG (Mean Teacher) ──
_C.RPPG_SEMI = CN()
_C.RPPG_SEMI.ENABLED = False
_C.RPPG_SEMI.UNLABELED_PATH = ""          # Single unlabeled NPY dir
_C.RPPG_SEMI.UNLABELED_PATHS = []        # Multi-source unlabeled NPY dirs
_C.RPPG_SEMI.EMA_DECAY = 0.999            # Teacher EMA decay
_C.RPPG_SEMI.LAMBDA_UNSUP = 1.0           # Max unsupervised loss weight
_C.RPPG_SEMI.RAMP_UP_EPOCHS = 30          # Sigmoid ramp-up duration
_C.RPPG_SEMI.FREQ_WEIGHT = 0.5            # FrequencyConstraint weight (alpha)
_C.RPPG_SEMI.UNLABELED_BATCH_SIZE = 4
_C.RPPG_SEMI.EPOCHS = 80
_C.RPPG_SEMI.LR = 3e-3
_C.RPPG_SEMI.PATIENCE = 40

# ── Inference ──
_C.INFERENCE = CN()
_C.INFERENCE.FACE_CONFIDENCE_THRESHOLD = 0.4
_C.INFERENCE.USE_FP16 = False
_C.INFERENCE.WEBCAM_ID = 0
_C.INFERENCE.WEBCAM_W = 640
_C.INFERENCE.WEBCAM_H = 480
_C.INFERENCE.RPPG_BUFFER_SIZE = 128
_C.INFERENCE.RPPG_PREDICT_INTERVAL = 30
_C.INFERENCE.TARGET_FPS = 0
_C.INFERENCE.DISPLAY_WINDOW = True
_C.INFERENCE.SAVE_VIDEO = ""

# ── Output ──
_C.OUTPUT = CN()
_C.OUTPUT.CHECKPOINT_DIR = "checkpoints"
_C.OUTPUT.LOG_DIR = "logs"


def get_cfg_defaults():
    """Get a copy of the default config."""
    return _C.clone()


def get_config(config_file: str = None, opts: list = None):
    """Load config from YAML file with optional CLI overrides."""
    cfg = get_cfg_defaults()
    if config_file:
        with open(config_file, 'r', encoding='utf-8') as f:
            cfg.merge_from_other_cfg(cfg.load_cfg(f))
    if opts:
        cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg
