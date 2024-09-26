from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import final_inference as fi
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random  # This is just for the placeholder logic
import uvicorn

app = FastAPI()

# Serve the HTML file and static resources
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class QuestionPair(BaseModel):
    id1: int
    id2: int
    question1: str
    question2: str

def predict(data: QuestionPair):
    data = data.dict()
    
    c1 = data['id1']
    c2 = data['id2']
    c3 = data['question1']
    c4 = data['question2']
    # Call the prediction function
    dt_pred, dt_pred_proba = fi.inferencing(c1, c2, c3, c4)

    if int(dt_pred[0]) == 0:
        out = 'The Question pairs are Different '
        prob = dt_pred_proba[0][0]
    else:
        out = 'The Question pairs are Similar'
        prob = dt_pred_proba[0][1]

    return out, prob

class DuplicacyResult(BaseModel):
    isDuplicate: str
    probability: float

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return FileResponse("static/question_checker.html")

@app.post("/api/check-duplicacy", response_model=DuplicacyResult)
async def check_duplicacy(question_pair: QuestionPair):
    try:
        # Placeholder for duplicate detection logic
        # In a real-world scenario, you'd implement your ML model here
        is_duplicate, probability  = predict(question_pair)

        return DuplicacyResult(isDuplicate=is_duplicate, probability=probability)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
