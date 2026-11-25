from fastapi import FastAPI

from truyen_reco.api.routers import recommend

app = FastAPI(title="Truyen Reco API")
app.include_router(recommend.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
