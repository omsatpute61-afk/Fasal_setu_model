import pandas as pd
import sqlite3
import os

# 1. Define where your CSV is, and where the DB should go
csv_path = 'src/data/pestopia_treatments.csv'
db_path = 'src/export/advisory_database.db'

# Make sure the export folder actually exists
os.makedirs('src/export', exist_ok=True)

# 2. Run the conversion
if os.path.exists(csv_path):
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Converting to SQLite database for Flutter...")
    conn = sqlite3.connect(db_path)
    df.to_sql('treatments', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ Success! Your file is ready at: {db_path}")
else:
    print(f"❌ Error: Could not find {csv_path}. Make sure you are in the root Vision folder.")