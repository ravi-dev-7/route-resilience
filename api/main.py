"""
main.py
FastAPI application entry point for Route Resilience backend.
Connects segmentation model + graph logic to HTTP endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import segmentation
from api.routes import graph
from api.routes import simulation

app = FastAPI(
    title="Route Resilience API",
    description="Occlusion-robust road extraction + topological resilience analysis",
    version="1.0.0",
)

# allow frontend (Streamlit/React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router, prefix="/api", tags=["Segmentation"])
app.include_router(graph.router, prefix="/api", tags=["Graph"])
app.include_router(simulation.router, prefix="/api", tags=["Simulation"])
@app.get("/")
def root():
    return {"status": "Route Resilience API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)