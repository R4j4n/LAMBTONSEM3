import sqlite3
from typing import Optional


class SQLiteSchemaExtractor:
    """
    A class to extract the schema from an SQLite database file and return it as formatted text.
    """

    def __init__(self, db_path: str):
        """
        Initialize the extractor with the path to the SQLite database file.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None

    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return False

    def close(self):
        """Close the database connection if it exists."""
        if self.connection:
            self.connection.close()

    def get_schema(self) -> Optional[str]:
        """
        Extract the schema from the SQLite database and return it as formatted text.

        Returns:
            A string containing the formatted schema, or None if an error occurred
        """
        if not self.connect():
            return None

        try:
            cursor = self.connection.cursor()

            # Get the list of all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            schema_text = ""

            for table in tables:
                table_name = table[0]

                # Skip SQLite internal tables
                if table_name.startswith("sqlite_"):
                    continue

                # Get the CREATE TABLE statement for the current table
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()

                # Format the CREATE TABLE statement
                schema_text += f"CREATE TABLE {table_name} (\n"

                for i, col in enumerate(columns):
                    # Column format: column_name data_type
                    col_id, col_name, col_type, not_null, default_val, pk = col

                    # Map SQLite types to our simplified type system
                    simplified_type = self._simplify_type(col_type.lower())

                    # Add column definition
                    schema_text += f"    {col_name} {simplified_type}"

                    # Add comma if not the last column
                    if i < len(columns) - 1:
                        schema_text += ","

                    schema_text += "\n"

                schema_text += ")\n\n"

            # Remove the last newline
            if schema_text.endswith("\n\n"):
                schema_text = schema_text[:-1]

            return schema_text

        except sqlite3.Error as e:
            print(f"Error extracting schema: {e}")
            return None
        finally:
            self.close()

    def _simplify_type(self, sqlite_type: str) -> str:
        """
        Convert SQLite types to simplified types (number, text, others).

        Args:
            sqlite_type: The SQLite data type

        Returns:
            A simplified type name
        """
        if any(
            num_type in sqlite_type
            for num_type in ["int", "real", "floa", "doub", "num", "dec"]
        ):
            return "number"
        elif any(
            text_type in sqlite_type
            for text_type in ["text", "char", "clob", "varchar"]
        ):
            return "text"
        else:
            return "others"


# # Example usage:
# if __name__ == "__main__":
#     # Replace with your SQLite database file path
#     db_file = "path/to/your/database.db"

#     extractor = SQLiteSchemaExtractor(db_file)
#     schema = extractor.get_schema()

#     if schema:
#         print(schema)
#     else:
#         print("Failed to extract schema.")
