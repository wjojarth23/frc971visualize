from flask import Flask
import psycopg2
import pandas as pd

app = Flask(__name__)
DB_HOST = "scouting.frc971.org"
DB_PORT = 5000
DB_NAME = "postgres"
DB_USER = "tableau"
DB_PASSWORD = "xWYNKBkaHasO"

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

# Define routes
@app.route("/")
def index():
    return "Hello World!"

@app.route("/data")
def get_data():
    return df.to_html()

if __name__ == "__main__":
    app.run()
