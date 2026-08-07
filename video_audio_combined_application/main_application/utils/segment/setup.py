import sqlite3

# Creates the local database file named ddbb.db
conn = sqlite3.connect("ddbb.db")
cursor = conn.cursor()

# Creates a basic user table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

conn.commit()
conn.close()
print("Database 'ddbb.db' created successfully!")