
import os
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager
from typing import Callable
import requests

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)

MODEL_URI = "models:/iris-classifier@champion"


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


def load_model_version():
    client = mlflow.MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )
    champion = client.get_model_version_by_alias(
        name="iris-classifier",
        alias="champion"
    )
    return champion.version


def create_app(
    model_loader: Callable = load_mlflow_model,
    version_loader: Callable = load_model_version
):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MLflow champion model..\n{e}"
            ) from e

        app.state.model_version = version_loader()

        print(
            f"Loaded iris-classifier version "
            f"{app.state.model_version}"
        )

        yield

    app = FastAPI(
        title="Iris Model API",
        description="API for serving the Iris classification model",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.get("/")
    def health_check_root():
        return {"message": "API is ready to serve"}

    @app.get("/health")
    def health_check(request: Request):
        return {
            "status": "healthy",
            "model_name": "iris-classifier",
            "model_version": request.app.state.model_version
        }

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):
        model = request.app.state.model

        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]]

        prediction = int(model.predict(features)[0])

        return {
            "prediction": prediction,
            "class_name": [
                "setosa",
                "versicolor",
                "virginica"
            ][prediction]
        }

    return app


app = create_app()
