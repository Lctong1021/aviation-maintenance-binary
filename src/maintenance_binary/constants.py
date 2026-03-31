from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_NAME = "2days"
BENCHMARK_URL = "https://drive.google.com/uc?id=1-2pxwiQNhFnhTg7whosQoF_yztD5jOM2"

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "stage1"
LOCAL_BENCHMARK_DIR = PROJECT_ROOT / BENCHMARK_NAME

MAX_CHANNELS = 23
RANDOM_SEED = 42
