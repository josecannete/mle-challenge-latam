from contextlib import asynccontextmanager
import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field

from challenge.model import DelayModel
from challenge.constants import _VALID_OPERA_VALUES

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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: fastapi.Request, exc: RequestValidationError
):
    return JSONResponse(
        status_code=fastapi.status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


class Flight(BaseModel):
    OPERA: str = Field(pattern=r"|".join(_VALID_OPERA_VALUES))
    TIPOVUELO: str = Field(pattern=r"N|I")
    MES: int = Field(ge=1, le=12)


class BatchPredictionQuery(BaseModel):
    flights: list[Flight] = Field(min_items=1)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(query: BatchPredictionQuery) -> dict:
    data_df = pd.DataFrame.from_records(
        [flight.model_dump() for flight in query.flights]
    )
    features = delay_model.preprocess(data_df)
    predictions = delay_model.predict(features)
    return {"predict": predictions}
