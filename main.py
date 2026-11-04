import os
from pathlib import Path
from flask import app
import joblib
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel,Field
import mlflow
import mlflow.sklearn
from contextlib import asynccontextmanager
from typing import Callable
import requests
# app=FastAPI(title="Iris Model API")
MLFLOW_TRACKING_URI=os.getenv(
    "MLFLOW_TRACKING_URI","http://127.0.0.1:6001"
)
# model_path=Path("models/iris-model.joblib")
# model =joblib.load(model_path)

MODEL_URI="models:/iris-classifier@champion"


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)

class IrisRequest(BaseModel):
    sepal_length: float=Field(gt=0)
    sepal_width: float=Field(gt=0)
    petal_length: float=Field(gt=0)
    petal_width: float=Field(gt=0)

def create_app(model_loader: Callable = load_mlflow_model,version_loader:Callable=None):
    if version_loader():
        def version_loader():
            client=mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            champion = client.get_model_version_by_alias(
                name="iris-classifier",
                alias="champion",
            )
            return champion.version
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # load the model once during startup
        # Injected loader can be real mlflow or a test model
        try:
            app.state.model = model_loader()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MLflow champion model..\n{e}"
            ) from e

        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        champion = client.get_model_version_by_alias(
            name="iris-classifier",
            alias="champion",
        )
        app.state.model_version = version_loader()
        print(f"Loaded iris-classifier version {app.state.model_version}")
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
def health_check():
    return {"status":"healthy","model_name":"iris-classifier","model_version":app.state.model_version}

@app.post("/predict")
def predict(data:IrisRequest, request: Request):
    model=request.app.state.model
    features=[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction=int(app.state.model.predict(features)[0])
    return {"prediction":prediction,
            "class_name":["sentosa","versicolor","virginica"][prediction]
        }
