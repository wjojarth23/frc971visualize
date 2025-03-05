from flask import Flask, render_template_string
import psycopg2
import pandas as pd
import pygwalker as pyg

app = Flask(__name__)

# Database credentials
DB_HOST = "scouting.frc971.org"
DB_PORT = 5000
DB_NAME = "postgres"
DB_USER = "tableau"
DB_PASSWORD = "xWYNKBkaHasO"

def fetch_data():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    query = """
    SELECT * FROM stats2025 s
    LEFT JOIN team_match2025 m
        ON m.team_number = s.team_number
        AND m.match_number = s.match_number
        AND m.comp_code = s.comp_code
    WHERE s.comp_code IN ('2025camb', '2016nytr')
    ORDER BY s.match_number;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@app.route("/")
def index():
    df = fetch_data()
    pyg_html = pyg.walk(df, return_html=True)
    return render_template_string("<html><body>{{ html_content | safe }}</body></html>", html_content=pyg_html)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
