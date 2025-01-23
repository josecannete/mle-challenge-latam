from contextlib import asynccontextmanager
import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import DelayModel

delay_model = None


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global delay_model
    # Load the ML model
    delay_model = DelayModel()
    delay_model.load_model("trained_model.joblib")
    yield
    # Clean up the ML models and release the resources
    delay_model = None


app = fastapi.FastAPI(lifespan=lifespan)


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class BatchPredictionQuery(BaseModel):
    flights: list[Flight]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(query: BatchPredictionQuery) -> dict:
    data_df = pd.DataFrame.from_records([flight.dict() for flight in query.flights])
    features = delay_model.preprocess(data_df)
    predictions = delay_model.predict(features)
    return {"predict": predictions}
