import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
_LOCAL_ENV = _PROJECT_ROOT / '.env'
if _LOCAL_ENV.is_file():
    load_dotenv(_LOCAL_ENV)


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


MYAI_DIR = _env_path('MYAI_HOME', Path.home() / '.myai')
DATABASE_PATH = MYAI_DIR / 'db.sqlite'
SPACES_ROOT = MYAI_DIR / 'spaces'
SANDBOXES_ROOT = MYAI_DIR / 'sandboxes'


def use_dir(directory: Path) -> Path:
    """Point the application at an isolated myAI directory, e.g. in tests."""
    global MYAI_DIR, DATABASE_PATH, SPACES_ROOT, SANDBOXES_ROOT
    MYAI_DIR = Path(directory).expanduser().resolve()
    DATABASE_PATH = MYAI_DIR / 'db.sqlite'
    SPACES_ROOT = MYAI_DIR / 'spaces'
    SANDBOXES_ROOT = MYAI_DIR / 'sandboxes'
    return MYAI_DIR
