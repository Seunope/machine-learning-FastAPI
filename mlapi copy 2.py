
from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import api
from pydantic import BaseModel 
import pickle
import pandas as pd

app = FastAPI()

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "http://localhost:8001"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

origins = [
    "https://teste.com.br",
    "http://localhost:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api)

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

