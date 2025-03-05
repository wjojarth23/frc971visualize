from flask import Flask
import psycopg2
import pandas as pd
app = Flask(__name__)
DB_HOST = "scouting.frc971.org"
DB_PORT = 5000
DB_NAME = "postgres"
DB_USER = "tableau"       # your username here
DB_PASSWORD = "xWYNKBkaHasO"   # your password here

# Establish connection
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Define your SQL query.
query = """
SELECT *
FROM stats2025 s
WHERE s.comp_code IN ('2025camb', '2016nytr')
ORDER BY s.match_number;
"""

# Execute the query and load data into a pandas DataFrame
df = pd.read_sql_query(query, conn)

@app.route("/")
def hello():
    return "Hello World!"
@app.route("/data")
def hello():
    return df

if __name__ == "__main__":
    app.run()
