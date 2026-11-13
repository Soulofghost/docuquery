from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = str(DATA_DIR / "docuquery.db")
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 60
SIMILARITY_THRESHOLD = 0.12
EMBEDDING_DIM = 128
API_PREFIX = "/api/v1"
