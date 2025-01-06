import os

from dotenv import load_dotenv

load_dotenv("/home/r1j1n/Documents/Lambton/3304/.env")

api_key = os.getenv("API_KEY")

if api_key is None:
    raise ValueError("API_KEY not found in .env file")


print(api_key)
