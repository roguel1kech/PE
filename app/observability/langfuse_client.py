import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
env_path = ROOT_DIR / ".env"

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)

PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

if PUBLIC_KEY:
    os.environ["LANGFUSE_PUBLIC_KEY"] = PUBLIC_KEY
if SECRET_KEY:
    os.environ["LANGFUSE_SECRET_KEY"] = SECRET_KEY
if BASE_URL:
    os.environ["LANGFUSE_BASE_URL"] = BASE_URL

from langfuse import Langfuse

langfuse = Langfuse(
    public_key=PUBLIC_KEY,
    secret_key=SECRET_KEY,
    base_url=BASE_URL,
)
