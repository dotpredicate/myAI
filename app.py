from flask import Flask
import psycopg2
import os
import glob
from datetime import datetime
import hashlib
import binascii

app = Flask(__name__)

# Database connection
conn = psycopg2.connect(
    dbname='myai',
    user='myai',
    password='myai',
    host='localhost',
    port='5432'
)

# Initialize database schema
def init_database():
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS migrations ( 
                order_executed INTEGER GENERATED ALWAYS AS IDENTITY,
                name VARCHAR(255) NOT NULL,
                hash CHARACTER(64) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL
            )
        """)
    with conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS "migrations_name" ON "migrations" (name)
        """)

# Apply migrations
migrations_dir = 'migrations'

# Sort migration files lexicographically
migration_files = sorted(glob.glob(os.path.join(migrations_dir, '*.sql')))
for migration_file in migration_files:
    migration_name = os.path.basename(migration_file)
    # Check if migration already exists
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM migrations WHERE name = %s", (migration_name,))
        if cur.fetchone():
            print(f'Skipping already executed migration: {migration_name}')
            continue
    
    # Execute migration SQL
    with open(migration_file, 'r') as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    
    # Record migration execution
    start_time = datetime.now()
    hash_bytes = hashlib.sha256(migration_name.encode('utf-8')).digest()
    hash_str = binascii.hexlify(hash_bytes).decode('utf-8')
    end_time = datetime.now()
    
    with conn.cursor() as cur:
        cur.execute("INSERT INTO migrations (name, hash, start_time, end_time) VALUES (%s, %s, %s, %s)",
                   (migration_name, hash_str, start_time, end_time))
    conn.commit()
    print(f'Applied migration: {migration_file}')

@app.route('/')
def index():
    with conn.cursor() as cur:
        cur.execute('SELECT * FROM your_table')
    return str(cur.fetchall())

if __name__ == '__main__':
    app.run(debug=True)
    init_database()
