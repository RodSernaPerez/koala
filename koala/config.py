from pathlib import Path
import logging

logging = logging.getLogger(__name__)
DICT_MODELS = {"en": "distilbert-base-uncased",
               "es": "mrm8488/RuPERTa-base"}

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_FOLDER / "data"

MODELS_FOLDER = PROJECT_FOLDER / "models"
