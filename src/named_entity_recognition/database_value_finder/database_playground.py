import psycopg2

# Connect to an existing database

conn = psycopg2.connect(
    database="cordis",
    user="postgres",
    host="localhost",
    password="postgres"
)

# Open a cursor to perform database operations
cur = conn.cursor()

# Query the database and obtain data as Python objects
cur.execute("SELECT * FROM unics_cordis.project_programmes;")
result = cur.fetchone()
print(result)

# Close communication with the database
cur.close()
conn.close()
