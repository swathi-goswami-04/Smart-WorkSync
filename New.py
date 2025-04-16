import sqlite3

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# List of new columns (without UNIQUE/NOT NULL constraints)
new_columns = {
    "userid": "INTEGER",  # ✅ Remove UNIQUE NOT NULL
    "ph_no": "TEXT",
    "verify": "BOOLEAN DEFAULT 0",
    "joining": "DATETIME",
    "lastseen": "DATETIME",
    "status": "TEXT",
    "otp_secret": "TEXT",
    
}

# Check which columns already exist
cursor.execute("PRAGMA table_info(users);")
existing_columns = {col[1] for col in cursor.fetchall()}

# Add missing columns
for column, data_type in new_columns.items():
    if column not in existing_columns:
        alter_query = f"ALTER TABLE users ADD COLUMN {column} {data_type};"
        cursor.execute(alter_query)
        print(f"✅ Added missing column: {column}")

conn.commit()
conn.close()

print("✅ Database updated successfully!")




