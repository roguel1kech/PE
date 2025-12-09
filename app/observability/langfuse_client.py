import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

ROOT_DIR = Path(__file__).resolve().parents[2]
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

langfuse = Langfuse(
    public_key=PUBLIC_KEY,
    secret_key=SECRET_KEY,
    base_url=BASE_URL,
)
