import os
from dotenv import find_dotenv, load_dotenv
from logging.config import dictConfig


ROOT_ABS_DIR = os.path.abspath(os.path.dirname(__file__))

load_dotenv(find_dotenv())

FLUSONIC_TOKEN = os.environ.get('FLUSONIC_TOKEN')