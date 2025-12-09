from fastapi import FastAPI
from pydantic import BaseModel
from .models.sentiment_model import sentiment_analyzer

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(data: InputText):
    return sentiment_analyzer.predict(data.text)
