from pydantic import BaseModel

# Response returned by /predict endpoint
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float


# Health check response
class HealthResponse(BaseModel):
    message: str