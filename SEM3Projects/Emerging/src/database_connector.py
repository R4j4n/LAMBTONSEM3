import re
import sqlalchemy.exc
from typing import Literal
from typing import Iterable
from sqlalchemy import create_engine

from src.db_models import TableColumn, Table


class DatabaseAgent:

    def __init__(
        self,
        host_name: str,
        port: int,
        username: str,
        database_name: str,
        database_type=Literal["MySQL", "PostgreSQL", "MariaDB", "SQLite", "Oracle SQL"],
        password: str = None,
    ) -> None:
        self.__host_name: str = host_name
        self.__port: int = port
        self.__username: str = username
        self.__password: str = password
        self.__db_name: str = database_name
        self.db_type: str = database_type

        # check the connection to make sure database is there.
        self.__check_connection()

    def __generate_conn_str(self):
        """
        Generates connection string based on the type of database

        """
        conn_mapp = {
            "MySQL": f"mysql+pymysql://{self.__username}:{self.__password}@{self.__host_name}:{str(self.__port)}/{self.__db_name}",
            "PostgreSQL": f"postgresql+psycopg2://{self.__username}:{self.__password}@{self.__host_name}:{str(self.__port)}/{self.__db_name}",
        }

        self.conn_str = conn_mapp.get(self.db_type)

        if self.conn_str is None:

            raise NotImplementedError(f"Not implemented for {self.db_type}.")

        else:

            return self.conn_str

    def __check_connection(self):

        # get the connection string
        connection_str = self.__generate_conn_str()

        engine = create_engine(connection_str)

        try:
            engine.connect()
            print("Connection to MySQL database successful!")

        except Exception as e:

            raise f"Error connecting to MySQL database: {e}"

    def grab_table_names(self) -> Iterable:
        """Grabs all the table names in the database.

        Returns:
            Iterable: The names of the tables.
        """

        # get the database url
        url = self.__generate_conn_str()

        # create the engine
        engine = create_engine(url=url)

        # Grab the table names from sql alchemy
        table_names = engine.table_names()

        # delete the table names after tables have been extracted
        engine.dispose()

        return table_names

    def grab_table_schema(self, tables: Iterable) -> Iterable[Table]:

        # store the schemas
        schemas = []

        # get the database url
        url = self.__generate_conn_str()

        # create the engine
        engine = create_engine(url=url)

        # iterate over the tables
        for table in tables:
            # store the columns data
            columns = []

            if self.db_type == "SQLite":
                sql = f"PRAGMA table_info({table});"
                # TODO : Work for SQLITE

            else:
                sql = f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table}';
                """

            # grab the schema from the sql query.
            schema = engine.execute(sql).fetchall()

            # iterate over the schemas
            for column_name, d_type in schema:
                columns.append(TableColumn(name=column_name, dtype=d_type))

            # save the schemas in schemas directory.
            schemas.append(Table(name=table, columns=columns))

        # return the schemas
        return schemas
