from fastapi import FastAPI
from pydantic import BaseModel
from models.llm_client import generate_llm

app = FastAPI(title="LLM API")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(data: Prompt):
    return generate_llm(data.prompt)
