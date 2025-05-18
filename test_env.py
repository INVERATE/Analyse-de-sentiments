from dotenv import load_dotenv
import os

load_dotenv()  # charge les variables de .env
token = os.getenv("HF_TOKEN")
print(token)
