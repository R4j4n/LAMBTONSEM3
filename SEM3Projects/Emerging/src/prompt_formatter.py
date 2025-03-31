from typing import Any, Iterable
from src.db_models import Table


class PromptFormatterV1:

    def __init__(self, tables: Iterable[Table], db_type: str) -> None:
        """Format schema and question into a proper prompt.

        Args:
            table (Iterable[Table]): The list of Table object.
            db_type (str) : The name of the database type.
        """
        self.tables = tables
        self.db_type = db_type

    def format(self, table: Table) -> str:
        """Get table format."""
        table_fmt = []
        table_name = table.name
        for col in table.columns or []:
            # This is technically an incorrect type, but it should be a catchall word
            table_fmt.append(f"    {col.name} {col.dtype or 'any'}")
        if table.pks:
            table_fmt.append(
                f"    primary key ({', '.join(pk.name for pk in table.pks)})"
            )
        for fk in table.fks or []:
            table_fmt.append(
                f"    foreign key ({fk.column.name}) references {fk.references_name}({fk.references_column.name})"  # noqa: E501
            )
        if table_fmt:
            all_cols = ",\n".join(table_fmt)
            create_tbl = f"CREATE TABLE {table_name} (\n{all_cols}\n)"
        else:
            create_tbl = f"CREATE TABLE {table_name}"
        return create_tbl

    def __call__(self, question: str | None = None) -> str:

        temp = []
        for table in self.tables:
            temp.append(self.format(table))

        # add temp table info to the main prompt:
        main_prompt = "\n\n".join(temp)

        # sql prefix to start the generation
        sql_prefix = "SELECT"

        # return the final prompt.
        return f"""{main_prompt}\n\n\n-- Using valid {self.db_type}, answer the following questions for the tables provided above.\n\n-- {question}\n{sql_prefix}"""
