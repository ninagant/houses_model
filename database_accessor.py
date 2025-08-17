import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

class DatabaseAccessor:
    def __init__(self):
        load_dotenv()
        self.connection = psycopg2.connect(
            host=os.getenv("DB_HOSTNAME"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USERNAME"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        self.cursor = self.connection.cursor()

    def get_all_elements(self):
        query = "SELECT * FROM listings"
        df = pd.read_sql(query, self.connection)
        self.connection.close()
        return df

if __name__ == "__main__":
    db_accessor = DatabaseAccessor()
    # Perform database operations
    df = db_accessor.get_all_elements()
    print(df.head())