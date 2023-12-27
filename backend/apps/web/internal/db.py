from peewee import *

DB = SqliteDatabase("./data/ollama.db")
DB.connect()
