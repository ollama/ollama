from peewee import *

DB = SqliteDatabase("./ollama.db")
DB.connect()
