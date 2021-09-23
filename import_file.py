from multiprocessing import freeze_support
from tqdm import tqdm
from colorama import init, Fore, Back, Style
import numpy as np
import pandas as pd
import os
import MySQLdb
import itdbloader

# 경로와 포맷 지정
ROOT_PATH = os.getcwd()
DB_CONFIG = "~/.itdb.cnf"


def init_db(c, db_name):
    """Create itdb database

    requires: root username and pw in mysql_default.json
    Args:
        db_name: name of database
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
    Args:
        c: cursor
        db_name: name of database
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


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[0] == "y":
            return True
        if reply[0] == "n":
            return False


def main():
    init(autoreset=True)
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

    init_db(c, db_name)
    flag = create_table(c, db_name)
    c.close()           # close connection to open new one in itdbloader
    if flag is False:   # if create_table failed
        return

    itdbloader.main()


if __name__ == "__main__":
    freeze_support()
    main()
