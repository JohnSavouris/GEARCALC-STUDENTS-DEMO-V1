from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import sys
import subprocess
import threading
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# PATHS (relative to this file / repo root)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
NN_DIR = BASE_DIR / "NN_model"
NN_SCRIPT = NN_DIR / "NN_single_call.py"

FEATURES_CSV = NN_DIR / "features.csv"
XVALS_CSV = NN_DIR / "x_vals.csv"
YVALS_CSV = NN_DIR / "y_vals.csv"

# IMPORTANT:
# Το NN_single_call.py γράφει/διαβάζει fixed filenames μέσα στο NN_model/.
# Άρα για demo κάνουμε serialize τα requests για να μην "πατήσουν" το ένα το άλλο.
NN_RUN_LOCK = threading.Lock()


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI(
    title="GEARCALC-PRO NN API",
    version="0.1.0",
    description="Backend bridge for NN-based STE prediction (MATLAB/Python NN wrapper style)."
)

# CORS: για να κάνει fetch το GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # για demo. αργότερα βάλε το GitHub Pages domain σου
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request model (defaults = MATLAB script defaults)
# ============================================================
class STERequest(BaseModel):
    x_vals_pop: int = Field(1000, ge=10, le=50000, description="Points in STE curve")
    sy: float = Field(1000.0, gt=0)

    Z1: int = Field(50, ge=3, le=1000)
    Z2: int = Field(50, ge=3, le=1000)

    Cc1: float = Field(0.15)
    Cc2: float = Field(0.15)
    Ck1: float = Field(1.0)
    Ck2: float = Field(1.0)
    Cf1: float = Field(1.0)
    Cf2: float = Field(1.0)

    # Στο frontend βολεύει σε deg. Στο backend το γυρνάμε σε rad.
    a0_deg: float = Field(20.0, gt=0.0, lt=89.0)

    Cs1: float = Field(0.49)
    Cs2: float = Field(0.49)

    E_to_sy: float = Field(200.0, gt=0.0)
    ni: float = Field(0.3, ge=0.0, lt=0.5)

    log_Tmid_to_max: float = Field(-0.5)
    b_to_m: float = Field(30.0, gt=0.0)
    da12_to_m: float = Field(0.0)

    m: float = Field(3.0, gt=0.0)

    debug: bool = Field(False)


# ============================================================
# Helpers
# ============================================================
def to_json_scalar(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def to_json_list(arr: np.ndarray) -> List[Optional[float]]:
    arr = np.asarray(arr).reshape(-1)
    out = []
    for v in arr:
        try:
            x = float(v)
            out.append(x if math.isfinite(x) else None)
        except Exception:
            out.append(None)
    return out


def ensure_files_exist():
    if not NN_DIR.exists():
        raise FileNotFoundError(f"Missing NN_model folder: {NN_DIR}")
    if not NN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing NN script: {NN_SCRIPT}")
    # Δεν ελέγχω όλα τα αρχεία ένα-ένα γιατί μπορεί να έχεις παραλλαγές.
    # Αλλά αν θες, μπορούμε να βάλουμε και explicit checks εδώ.


def write_csv_inputs(features: np.ndarray, x_vals: np.ndarray):
    NN_DIR.mkdir(parents=True, exist_ok=True)
    # MATLAB style: row vectors
    np.savetxt(FEATURES_CSV, features.reshape(1, -1), delimiter=",", fmt="%.15g")
    np.savetxt(XVALS_CSV, x_vals.reshape(1, -1), delimiter=",", fmt="%.15g")


def run_nn_single_call() -> Dict[str, Any]:
    """
    Τρέχει το NN_model/NN_single_call.py με:
      python NN_model/NN_single_call.py
    και cwd = repo root (BASE_DIR)

    Γιατί cwd=BASE_DIR;
    - Το script σου έχει path_to_nn_model = 'NN_model'
    - Άρα περιμένει να υπάρχει φάκελος NN_model/ δίπλα (στο repo root)
    """
    cmd = [sys.executable, str(NN_SCRIPT)]

    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )

    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "cmd": cmd,
        "cwd": str(BASE_DIR),
    }


def read_y_vals() -> np.ndarray:
    if not YVALS_CSV.exists():
        raise FileNotFoundError(f"NN output not found: {YVALS_CSV}")

    y = np.genfromtxt(YVALS_CSV, delimiter=",")
    y = np.asarray(y).reshape(-1)
    return y


# ============================================================
# Core compute (MATLAB wrapper logic port)
# ============================================================
def compute_ste(req: STERequest) -> Dict[str, Any]:
    ensure_files_exist()

    # ---------------------------
    # Parse inputs (MATLAB defaults)
    # ---------------------------
    x_vals_pop = int(req.x_vals_pop)
    sy = float(req.sy)

    Z1 = int(req.Z1)
    Z2 = int(req.Z2)
    Cc1 = float(req.Cc1)
    Cc2 = float(req.Cc2)
    Ck1 = float(req.Ck1)
    Ck2 = float(req.Ck2)
    Cf1 = float(req.Cf1)
    Cf2 = float(req.Cf2)
    a0 = math.radians(float(req.a0_deg))
    Cs1 = float(req.Cs1)
    Cs2 = float(req.Cs2)
    E_to_sy = float(req.E_to_sy)
    ni = float(req.ni)
    log_Tmid_to_max = float(req.log_Tmid_to_max)
    b_to_m = float(req.b_to_m)
    da12_to_m = float(req.da12_to_m)
    m = float(req.m)

    # ---------------------------
    # Same pre/post calculations as MATLAB script
    # ---------------------------
    da12 = m * da12_to_m
    a12 = (Z1 + Z2) * m / 2.0 + da12  # currently not used in final outputs, but kept
    E1 = sy * E_to_sy
    E2 = sy * E_to_sy
    E = 2.0 * E1 * E2 / (E1 + E2)

    ni1 = ni
    ni2 = ni
    b = b_to_m * m
    b1 = b
    b2 = b

    # Feature vector (EXACT order as MATLAB script)
    features = np.array([
        Z1, Z2, Cc1, Cc2, Ck1, Ck2, Cf1, Cf2, a0, Cs1, Cs2,
        E_to_sy, ni, log_Tmid_to_max, b_to_m, da12_to_m, m
    ], dtype=float)

    # x values
    x_vals = np.linspace(0.0, 1.0, x_vals_pop, dtype=float)

    # ---------------------------
    # Critical section (shared CSV files)
    # ---------------------------
    with NN_RUN_LOCK:
        write_csv_inputs(features, x_vals)

        # Προαιρετικά: σβήσε παλιό y_vals για να μη διαβάσεις stale output αν το script σκάσει
        if YVALS_CSV.exists():
            try:
                YVALS_CSV.unlink()
            except Exception:
                pass

        run_info = run_nn_single_call()

        if run_info["returncode"] != 0:
            raise RuntimeError(
                "NN_single_call.py failed.\n"
                f"Return code: {run_info['returncode']}\n"
                f"STDOUT:\n{run_info['stdout']}\n"
                f"STDERR:\n{run_info['stderr']}"
            )

        y_vals = read_y_vals()

    # ---------------------------
    # MATLAB post-processing
    # ---------------------------
    STE_bar = y_vals

    rg1 = m * Z1 / 2.0 / math.cos(a0)
    rg2 = m * Z2 / 2.0 / math.cos(a0)

    Tmid_max1 = sy * (m**3) * b_to_m * (Z1/2 + Ck1) * (((Z1/2 - Cf1) * math.pi)**2) / (24.0 * (Z1**2) * (Ck1 + Cf1))
    Tmid_max2 = sy * (m**3) * b_to_m * (Z2/2 + Ck2) * (((Z2/2 - Cf2) * math.pi)**2) / (24.0 * (Z2**2) * (Ck2 + Cf2))
    Tmid = (10.0 ** log_Tmid_to_max) * math.sqrt(Tmid_max1 * Tmid_max2)

    T2 = Tmid * math.sqrt(Z2 / Z1)

    conv_factor_bar_to_rad = Tmid / (E * b * rg1 * rg2)

    STE_phi_mid = STE_bar * conv_factor_bar_to_rad
    STE_phi_2 = STE_phi_mid * math.sqrt(Z1 / Z2)

    # Avoid division by zero
    G = np.full_like(STE_phi_2, np.nan, dtype=float)
    mask_nonzero = np.abs(STE_phi_2) > 1e-18
    G[mask_nonzero] = T2 / STE_phi_2[mask_nonzero]

    # ---------------------------
    # Summary values (useful for frontend badges)
    # ---------------------------
    def nan_safe_stat(fn, arr):
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return float(fn(finite))

    summary = {
        "STE_bar_min": nan_safe_stat(np.min, STE_bar),
        "STE_bar_max": nan_safe_stat(np.max, STE_bar),
        "STE_bar_mean": nan_safe_stat(np.mean, STE_bar),
        "STE_bar_rms": nan_safe_stat(lambda a: np.sqrt(np.mean(a*a)), STE_bar),

        "STE_phi_2_min": nan_safe_stat(np.min, STE_phi_2),
        "STE_phi_2_max": nan_safe_stat(np.max, STE_phi_2),
        "STE_phi_2_mean": nan_safe_stat(np.mean, STE_phi_2),

        "G_min": nan_safe_stat(np.min, G),
        "G_max": nan_safe_stat(np.max, G),
        "G_mean": nan_safe_stat(np.mean, G),
    }

    response = {
        "ok": True,

        # Echo inputs (useful for reproducibility)
        "inputs": {
            "x_vals_pop": x_vals_pop,
            "sy": sy,
            "Z1": Z1, "Z2": Z2,
            "Cc1": Cc1, "Cc2": Cc2,
            "Ck1": Ck1, "Ck2": Ck2,
            "Cf1": Cf1, "Cf2": Cf2,
            "a0_deg": float(req.a0_deg),
            "a0_rad": a0,
            "Cs1": Cs1, "Cs2": Cs2,
            "E_to_sy": E_to_sy,
            "ni": ni,
            "log_Tmid_to_max": log_Tmid_to_max,
            "b_to_m": b_to_m,
            "da12_to_m": da12_to_m,
            "m": m
        },

        # Scalars from MATLAB wrapper
        "scalars": {
            "da12": to_json_scalar(da12),
            "a12": to_json_scalar(a12),
            "E": to_json_scalar(E),
            "b": to_json_scalar(b),
            "rg1": to_json_scalar(rg1),
            "rg2": to_json_scalar(rg2),
            "Tmid_max1": to_json_scalar(Tmid_max1),
            "Tmid_max2": to_json_scalar(Tmid_max2),
            "Tmid": to_json_scalar(Tmid),
            "T2": to_json_scalar(T2),
            "conv_factor_bar_to_rad": to_json_scalar(conv_factor_bar_to_rad)
        },

        # Curves for plotting in frontend
        "curves": {
            "x_vals": to_json_list(x_vals),
            "STE_bar": to_json_list(STE_bar),
            "STE_phi_mid": to_json_list(STE_phi_mid),
            "STE_phi_2": to_json_list(STE_phi_2),
            "G": to_json_list(G)
        },

        "summary": summary
    }

    if req.debug:
        response["debug"] = {
            "python_executable": sys.executable,
            "base_dir": str(BASE_DIR),
            "nn_dir": str(NN_DIR),
            "nn_script": str(NN_SCRIPT),
            "run_info": run_info
        }

    return response


# ============================================================
# API endpoints
# ============================================================
@app.get("/")
def root():
    return {"ok": True, "service": "GEARCALC-PRO NN API", "version": "0.1.0"}


@app.get("/health")
def health():
    return {
        "ok": True,
        "python": sys.executable,
        "paths": {
            "base_dir": str(BASE_DIR),
            "nn_dir_exists": NN_DIR.exists(),
            "nn_script_exists": NN_SCRIPT.exists(),
            "features_csv_exists": FEATURES_CSV.exists(),
            "x_vals_csv_exists": XVALS_CSV.exists(),
            "y_vals_csv_exists": YVALS_CSV.exists(),
        }
    }


@app.post("/api/ste")
def api_ste(req: STERequest):
    try:
        return compute_ste(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{e}\n\n{tb}")