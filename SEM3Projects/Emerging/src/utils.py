import os
import re

import pandas as pd
from sqlalchemy import create_engine, text


class SQLExtractor:
    def __init__(self, text: str):
        self.text = text

    def extract_select_commands(self):
        # Regular expression to find all SELECT commands
        select_pattern = re.compile(r"SELECT\s.*?;", re.IGNORECASE | re.DOTALL)
        select_commands = select_pattern.findall(self.text)
        return select_commands


def get_data_from_query(query, db_url, params=None):
    engine = create_engine(db_url)
    query = text(query)
    with engine.connect() as connection:
        raw_conn = connection.connection
        data = pd.read_sql_query(str(query), raw_conn, params=params)
    engine.dispose()
    return data


def search_files(directory, search_string):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string in file:
                return os.path.join(root, file)


def extract_and_correct_sql(text, correct=False):
    lines = text.splitlines()

    start_index = 0
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("SELECT"):
            start_index = i
            break

    generated_sql = "\n".join(lines[start_index:])

    if correct:
        if not generated_sql.strip().endswith(";"):
            generated_sql = generated_sql.strip() + ";"

    return generated_sql
