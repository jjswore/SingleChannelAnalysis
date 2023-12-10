import psycopg2

def connect_to_db():
    try:
        psycopg2.connect(
            dbname="SingleChannelAnalysisDB",
            port=4321,
            user="postgres",
            password="D3f3ns357!",
            host="localhost"
        )
    except psycopg2.Error as e:
        print("I am unable to connect to the database.")
        print(e)
