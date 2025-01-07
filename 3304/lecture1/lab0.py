import os

from dotenv import load_dotenv
from openai import OpenAI


class OpenAiLocal:
    """
    A class to generate embeddings using OpenAI's API.
    """

    def __init__(
        self, api_key_env_var="OPENAI_API_KEY", model="text-embedding-ada-002"
    ):
        """
        Initializes the embedding generator.

        Args:
            api_key_env_var (str): The environment variable name containing the API key.
            model (str): The OpenAI model to use for generating embeddings.
        """
        self.api_key_env_var = api_key_env_var
        self.model = model
        self.api_key = self._load_api_key()
        self.client = OpenAI(
            api_key="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"
        )

    def _load_api_key(self):

        load_dotenv("/home/r1j1n/Documents/GitHub/LAMBTONSEM3/3304/.env")
        api_key = os.getenv(self.api_key_env_var)
        print(api_key)
        if not api_key:
            raise ValueError(
                f"API key not found in environment variable: {self.api_key_env_var}"
            )
        return api_key

    def generate_embedding(self, text):

        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    def generate_response(self, prompt):
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}]
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# Example usage
if __name__ == "__main__":
    generator = OpenAiLocal(api_key_env_var="API_KEY", model="gpt-3.5-turbo")


if __name__ == "__main__":
    print("Welcome to Your Personal ChatGPT Interface!")

    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() == "exit":
            print("Exiting Chat Interface")
            break

        response = generator.generate_response(user_prompt)

        if response:
            print(f"ChatGPT: {response}\n")
