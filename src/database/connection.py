# -- Source - https://stackoverflow.com/a/54785824
# -- Posted by sandeep shewale
# -- Retrieved 2026-08-27, License - CC BY-SA 4.0

import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
# connection = psycopg2.connect(database=os.getenv("DB_NAME"), 
#                               user=os.getenv("DB_USER"), 
#                               password=os.getenv("DB_PASSWORD"), 
#                               host=os.getenv("DB_HOST"), 
#                               port=os.getenv("DB_PORT"))

# Turned above into a function so that I can call it from other files
def get_connection():
    connection = psycopg2.connect(database=os.getenv("DB_NAME"), 
                              user=os.getenv("DB_USER"), 
                              password=os.getenv("DB_PASSWORD"), 
                              host=os.getenv("DB_HOST"), 
                              port=os.getenv("DB_PORT"))
    return connection


connection = get_connection()
cursor = connection.cursor()
print(os.getenv("DB_NAME"))

# Fetch all rows from database
#record = cursor.fetchall()

#print("Data from Database:- ", record)
