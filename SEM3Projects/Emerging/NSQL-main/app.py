import os
import tempfile

import gradio as gr
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from schema_extractor import SQLiteSchemaExtractor


# Load model and tokenizer
def load_model():
    config = PeftConfig.from_pretrained("Rajan/training_run")
    tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-350M")
    base_model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-350M")
    model = PeftModel.from_pretrained(base_model, "Rajan/training_run")
    return model, tokenizer


# Extract and correct SQL from generated text
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


# Function to handle file upload and schema extraction
def upload_and_extract_schema(db_file):
    if db_file is None:
        return "Please upload a database file", None

    try:
        # Get the file path directly from Gradio
        temp_db_path = db_file.name

        extractor = SQLiteSchemaExtractor(temp_db_path)
        schema = extractor.get_schema()
        return schema, temp_db_path
    except Exception as e:
        return f"Error extracting schema: {str(e)}", None


# Function to handle chat interaction with streaming effect
def generate_sql(question, schema, db_path, chat_history):
    if db_path is None or not schema:
        return (
            chat_history
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": "Please upload a database file first"},
            ],
            None,
        )

    try:
        # Load model
        model, tokenizer = load_model()

        # Format prompt
        prompt_format = """ 
{}
-- Using valid SQLite, answer the following questions for the tables provided above.
{}
SELECT"""

        # Format the prompt with schema and question
        prompt = prompt_format.format(schema, question)

        # Generate SQL
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=500)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract SQL
        sql_query = extract_and_correct_sql(generated_text, correct=True)

        # Update history using dictionary format
        new_history = chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql_query},
        ]
        return new_history, sql_query
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return (
            chat_history
            + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": error_message},
            ],
            None,
        )


# Function for streaming SQL generation effect
def stream_sql(question, schema, db_path, chat_history):
    # First add the user message to chat
    yield chat_history + [{"role": "user", "content": question}], ""

    if db_path is None or not schema:
        yield chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Please upload a database file first"},
        ], "Please upload a database file first"
        return

    try:
        # Load model
        model, tokenizer = load_model()

        # Format prompt
        prompt_format = """ 
{}
-- Using valid SQLite, answer the following questions for the tables provided above.
{}
SELECT"""

        # Format the prompt with schema and question
        prompt = prompt_format.format(schema, question)

        # Generate SQL
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generated_ids = model.generate(input_ids, max_length=500)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract SQL
        sql_query = extract_and_correct_sql(generated_text, correct=True)

        # Fixed medium speed (0.03 seconds delay)
        import time

        delay = 0.03  # 30ms - normal typing speed

        # Stream the SQL query character by character for effect
        partial_sql = ""
        for char in sql_query:
            partial_sql += char
            # Update chat history and SQL display with each character
            yield chat_history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": partial_sql},
            ], partial_sql
            time.sleep(delay)  # Medium speed typing effect

    except Exception as e:
        error_message = f"Error: {str(e)}"
        yield chat_history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_message},
        ], error_message


# Main application
def create_app():
    with gr.Blocks(title="SQL Query Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# SQL Query Generator")
        gr.Markdown(
            "Upload a SQLite database, ask questions, and get SQL queries automatically generated"
        )

        # Store database path
        db_path = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                file_input = gr.File(label="Upload SQLite Database (.db file)")
                extract_btn = gr.Button("Extract Schema", variant="primary")

                # Schema display
                schema_output = gr.Textbox(
                    label="Database Schema", lines=10, interactive=False
                )

            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Query Conversation", height=400, type="messages"
                )

                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question about your data",
                        placeholder="e.g., Show me the top 10 most sold items",
                    )
                    submit_btn = gr.Button("Generate SQL", variant="primary")

                # SQL output display
                sql_output = gr.Code(
                    language="sql", label="Generated SQL Query", interactive=False
                )

        # Event handlers
        extract_btn.click(
            fn=upload_and_extract_schema,
            inputs=[file_input],
            outputs=[schema_output, db_path],
        )

        submit_btn.click(
            fn=stream_sql,
            inputs=[question_input, schema_output, db_path, chatbot],
            outputs=[chatbot, sql_output],
            api_name="generate",
            queue=True,
        )

        # Also trigger on enter key
        question_input.submit(
            fn=stream_sql,
            inputs=[question_input, schema_output, db_path, chatbot],
            outputs=[chatbot, sql_output],
            api_name="generate_on_submit",
            queue=True,
        )

    return app


# Launch the app
if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
