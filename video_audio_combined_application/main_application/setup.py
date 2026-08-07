import sqlite3

# Connect to SQLite (opens ddbb.db)
conn = sqlite3.connect("ddbb.db")
cursor = conn.cursor()

# Create the user_register table that app.py is expecting
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_register (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT UNIQUE,
    name TEXT,
    email TEXT,
    password TEXT
)
""")

conn.commit()
conn.close()
print("Table 'user_register' created successfully!")