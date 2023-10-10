
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel 
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):  
    age:int
    location: str
    gender: str
    loanAmount: float
    loanTenor: int
    Female:int
    Male: int
    Married: int
    Widowed: int
    Single: int

# rb: read as binary
with open('modelRCoef.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def scoringEndpoint():
    return {"Hello": "World"}


@app.post("/send")
async def scoringEndpoint(item: ScoringItem):
    df = pd.DataFrame( [item.dict().values()], columns=item.dict().keys())
    yhat = model.predict(df)

    return {
        "prediction": int(yhat)
    }

