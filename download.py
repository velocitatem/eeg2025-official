import logging
from pathlib import Path
from datetime import datetime
from mne import get_config
from eegdash import EEGChallengeDataset
from joblib import Parallel, delayed

# --- Paths ---
CACHE_DIR = Path(get_config("MNE_DATA")).expanduser() / "eeg_challenge_bdf"
LOG_DIR = CACHE_DIR / "logs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"download_{datetime.now():%Y%m%d-%H%M%S}.log"

# --- Logging to file + console ---
fmt = "%(asctime)s | %(processName)s | %(levelname)s | %(name)s | %(message)s"
handlers = [
    logging.StreamHandler(),                           # console
    logging.FileHandler(log_file, encoding="utf-8"),   # file
]
logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, datefmt="%Y-%m-%d %H:%M:%S")
logging.captureWarnings(True)  # route warnings->logging
logger = logging.getLogger(__name__)
logger.info("Logging to %s", log_file)

# --- Dataset list ---
dataset_list = [
    EEGChallengeDataset(
        release=f"R{release}",
        cache_dir=CACHE_DIR,
        mini=False,
    )
    for release in range(1, 12)
]

def try_to_download(dataset):
    try:
        _ = dataset.raw
    except Exception as e:
        logging.getLogger(__name__).error("Failed to load raw for %s: %s", getattr(dataset, "s3file", dataset), e)

raws = Parallel(n_jobs=-1, prefer="threads")(
    delayed(try_to_download)(d) for dataset_obj in dataset_list for d in dataset_obj.datasets
)

logger.info("Done. Full log saved at: %s", log_file)
