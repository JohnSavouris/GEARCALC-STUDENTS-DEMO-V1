from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.ste_service import (
    FEATURE_NAMES,
    build_feature_vector,
    compute_ste_outputs,
    get_model,
    training_range_report,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_FILE = ROOT_DIR / "index.html"
LEGACY_HTML_FILE = ROOT_DIR / "GEARCALC-STUDENTS-DEMO-V3-01-03-2026.html"


class STEPredictRequest(BaseModel):
    z1: float = Field(..., gt=3)
    z2: float = Field(..., gt=3)
    cc1: float
    cc2: float
    ck1: float = Field(..., gt=0)
    ck2: float = Field(..., gt=0)
    cf1: float = Field(..., gt=0)
    cf2: float = Field(..., gt=0)
    a0_rad: float = Field(..., gt=0.0, lt=1.5)
    cs1: float
    cs2: float
    e_to_sy: float = Field(..., gt=1e-9)
    nu: float = Field(..., gt=0.0, lt=0.4999)
    log_tmid_to_max: float
    b_to_m: float = Field(..., gt=0)
    da12_to_m: float
    m: float = Field(..., gt=0)
    x_points: int = Field(1000, ge=64, le=4000)
    sy_ref: float = Field(1000.0, gt=0)


app = FastAPI(
    title="GEARCALC STE API",
    description="In-memory NN inference for STE integrated with GEARCALC.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    if HTML_FILE.exists():
        return FileResponse(HTML_FILE)
    if LEGACY_HTML_FILE.exists():
        return FileResponse(LEGACY_HTML_FILE)
    raise HTTPException(status_code=404, detail="Missing frontend file: index.html")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ste/meta")
def ste_meta() -> dict[str, object]:
    return {
        "feature_names": FEATURE_NAMES,
        "default_sy_ref": 1000.0,
        "x_domain": [0.0, 1.0],
    }


@app.post("/predict_ste")
def predict_ste(payload: STEPredictRequest) -> dict[str, object]:
    try:
        model_loaded = get_model() is not None
        features_dict = payload.model_dump(exclude={"x_points", "sy_ref"})
        features = build_feature_vector(features_dict)
        x_vals = np.linspace(0.0, 1.0, int(payload.x_points), dtype=float)
        result = compute_ste_outputs(features, x_vals, sy_ref=float(payload.sy_ref))
        range_report = training_range_report(features)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"STE prediction failed: {exc}") from exc

    out_of_range = [item for item in range_report if not item["in_range"]]
    return {
        "ok": True,
        "model_loaded": model_loaded,
        "features": features_dict,
        "range_report": range_report,
        "out_of_range_count": len(out_of_range),
        "results": result,
    }
