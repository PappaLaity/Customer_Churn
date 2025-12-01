import logging
import os
from logging.handlers import RotatingFileHandler

_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)
_log_dir = os.getenv("LOG_DIR", "logs")
_log_file = os.path.join(_log_dir, "api.log")

os.makedirs(_log_dir, exist_ok=True)

api_logger = logging.getLogger("api")
api_logger.setLevel(_level)

if not api_logger.handlers:
    _fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    _stream = logging.StreamHandler()
    _stream.setFormatter(_fmt)
    api_logger.addHandler(_stream)

    _rotating = RotatingFileHandler(_log_file, maxBytes=10_000_000, backupCount=3)
    _rotating.setFormatter(_fmt)
    api_logger.addHandler(_rotating)

api_logger.propagate = False
