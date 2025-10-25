import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PIPELINE_PATH = "pipeline_v1.bin"

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

app = FastAPI(title="Course Model API")

try:
    with open(PIPELINE_PATH,"rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {PIPELINE_PATH}: {e}")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(customer: Customer):
    try:
        proba= model.predict_proba([customer.__dict__])[0,1]
        return {"probability":float(proba)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))