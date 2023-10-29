
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel 
import pickle
import pandas as pd

app = FastAPI()

# Salt to your taste
ALLOWED_ORIGINS = '*'    # or 'foo.com', etc.

# handle CORS preflight requests
@app.options('/')
async def preflight_handler(request: Request, rest_of_path: str) -> Response:
    response = Response()
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response

# set CORS headers
@app.middleware("http")
async def add_CORS_header(request: Request, call_next):
    response = await call_next(request)
    response.headers['Access-Control-Allow-Origin'] = ALLOWED_ORIGINS
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization, Content-Type'
    return response

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

# @app.get("/")
# async def scoringEndpoint():
#     return {"Hello": "World"}


@app.post("/send")
async def scoringEndpoint(item: ScoringItem):
    df = pd.DataFrame( [item.dict().values()], columns=item.dict().keys())
    print('DD', df)
    yhat = model.predict(df)

    return {
        "repaymentCoefficient": str(int(yhat))+'%',
        "message": "This user has a "+str(int(yhat))+"% chance of repaying ₦"+ str(item.loanAmount)+" loan" 
    }

