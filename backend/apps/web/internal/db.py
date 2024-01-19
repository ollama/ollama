from peewee import *
from config import DATA_DIR


DB = SqliteDatabase(str(DATA_DIR / "ollama.db"))
DB.connect()
