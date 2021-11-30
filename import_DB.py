from multiprocessing import freeze_support
from colorama import init, Fore
import MySQLdb
import requests
import itdbloader
import pandas as pd
from tqdm import tqdm
from utils import yes_or_no
PATCH_URL="http://www.vloom.co.kr:3000/dev/music/interval"


def connect_db():
    print("Loading configuration...", end=" ")
    try:
        config = itdbloader.get_config()
        print(Fore.GREEN + "OK")

    except Exception as e:
        print("Please check if .itdb.config exists")
        return

    print("Checking MySQL connections...", end=" ")
    try:
        cnx = itdbloader.db_connect(config)
        print(Fore.GREEN + "OK")
    except MySQLdb.Error as e:
        print(Fore.RED + "failed")
        print(e)
        print("Please check if MySQL is running, or check the configuration file")
        return
    c = cnx.cursor()
    db_name = config.get("client", "database")

    return c, db_name


def init_db(c, db_name):
    """Create itdb database

    requires: root username and pw in mysql_default.json
    :param c: MySQLdb cursor
    :type c: MySQLdb cursor
    :param db_name: name of the database
    :type db_name: String
    :return: True if database created successfully
    :rtype: bool
    """
    print("Creating database...")
    result = c.execute(f"SHOW DATABASES LIKE '{db_name}';")
    # if db exists
    if result == 1:
        ans = yes_or_no("DB already exists. Do you want to overwrite?")
        if ans is True:
            ans2 = yes_or_no("Are you sure you want to overwrite?")
            if ans2 is True:
                c.execute(f"DROP DATABASE {db_name};")
            else:
                return
        else:
            return

    # create database
    try:
        c.execute(f"CREATE DATABASE {db_name}")
    except MySQLdb.Error as e:
        print(f"{Fore.RED} Failed to create database: {e} {Fore.RESET}")
        return
    print(f"Database created: {db_name}")
    return True


def create_table(c, db_name):
    """Create tracks and playlists table in database

    :param c: MySQLdb cursor
    :type c: MySQLdb cursor
    :param db_name: name of the database
    :type db_name: String
    :return: True if successful
    :rtype: bool
    """
    # create table
    print("Creating table...", end=" ")
    c.execute(f"USE {db_name};")
    try:
        query = open("./itdb.sql").read()
        c.execute(query)
        print(Fore.GREEN + "OK")
    except MySQLdb.Error as e:
        print(Fore.RED + "failed")
        print(e)
        return False
    return True

def send_patch(url, json_data):
    """send HTTP PATCH with given data

    :param data: csv data
    :return: status code, status message
    """
    r=requests.patch(url, json=json_data)
    # print(str(idx), str(value), r.json())
    return r.status_code, r.json()["status"]

def send_pitch(url, pitch_path):
    df_pitch=pd.read_csv(pitch_path, index_col=[0])
    df_pitch=df_pitch.fillna(df_pitch.mean())
    print(df_pitch.head())

    error_list = []
    for index, row in tqdm(df_pitch.iterrows(), total=df_pitch.shape[0]):
        data={"itunes_id": index, "frequency": round(row["mean"], 2)}
        status_code, success=send_patch(url, data)
        if not success:
            error_list.append(index)
            print(str(index), "Error code:"+ str(status_code), str(success))

def main():
    init(autoreset=True)
    c, db_name = connect_db()
    init_db(c, db_name)
    flag = create_table(c, db_name)
    c.close()  # close connection to open new one in itdbloader
    if flag is False:  # if create_table failed
        return

    # import to DB
    config = itdbloader.get_config()
    db_loader = itdbloader.load_itdb(config)


if __name__ == "__main__":
    freeze_support()
    main()
