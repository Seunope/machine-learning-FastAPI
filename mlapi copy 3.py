
from typing import Union

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 
import pickle
import pandas as pd



middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]


app = FastAPI(middleware=middleware)


class ScoringItem(BaseModel):  
    # gender: str
    # maritalStatus: str
    age:int
    # location: str
    loanAmount: float
    loanTenor: int
    female:int
    male: int
    Married: int
    Widowed: int
    Single: int

# rb: read as binary
with open('modelRCoef3.pkl', 'rb') as f:
    model = pickle.load(f)

@app.get("/")
async def scoringEndpoint():
    return {"Hello": "World"}


@app.post("/send")
async def scoringEndpoint(item: ScoringItem):
    df = pd.DataFrame( [item.dict().values()], columns=item.dict().keys())
    print('DD', df)
    yhat = model.predict(df)

    return {
        "repaymentCoefficient": str(int(yhat))+'%',
        "message": "This user has a "+str(int(yhat))+"% chance of repaying ₦"+ str(item.loanAmount)+" loan" 
    }

