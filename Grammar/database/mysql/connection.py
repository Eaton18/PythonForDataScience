
import mysql.connector
from mysql.connector import errorcode

config = {
    'user': 'root',
    'password': 'root',
    'host': '10.147.176.200',
    'port': 3306,
    'database': 'employees',
    'raise_on_warnings': True,
    'use_pure': True,
}


try:
    cnx = mysql.connector.connect(**config)
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    cnx.close()

