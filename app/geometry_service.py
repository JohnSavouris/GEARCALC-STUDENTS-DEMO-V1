from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe(v: float, lo: float) -> float:
    return float(max(lo, v))


def _linspace(a: float, b: float, n: int) -> np.ndarray:
    return np.linspace(float(a), float(b), int(max(2, n)), dtype=float)


def _inv_angle(alpha: float) -> float:
    return float(math.tan(alpha) - alpha)


def _inv_to_angle(invphi: float) -> float:
    if invphi < 0.0:
        raise ValueError("Invalid inv(): invphi < 0.")
    if invphi > math.pi:
        raise ValueError("Invalid inv(): invphi > pi.")
    if invphi == 0.0:
        return 0.0

    if invphi < 0.5:
        phi = 1.441 * (invphi ** (1.0 / 3.0)) - 0.366 * invphi
    else:
        phi = 0.243 * math.pi - 0.471 * math.atan(invphi)

    for _ in range(60):
        t = math.tan(phi)
        f = (t - phi) - invphi
        fp = t * t
        dphi = -f / max(1e-15, fp)
        phi += dphi
        if abs(dphi) <= 1e-13:
            break
    return float(phi)


def compute_operating_from_shifts(
    m: float,
    z1: int,
    z2: int,
    alpha0: float,
    ck: float,
    cf: float,
    x1: float,
    x2: float,
    n_lo: int = 400,
) -> dict[str, Any]:
    ro1 = 0.5 * m * z1
    ro2 = 0.5 * m * z2
    rb1 = ro1 * math.cos(alpha0)
    rb2 = ro2 * math.cos(alpha0)

    ra1 = ro1 + (ck + x1) * m
    ra2 = ro2 + (ck + x2) * m
    rf1 = ro1 - (cf - x1) * m
    rf2 = ro2 - (cf - x2) * m

    if not (ck + x1 > 0.0 and ck + x2 > 0.0):
        raise ValueError("Invalid: (Ck + x) must be > 0 for both gears.")
    if not (cf - x1 > 0.0 and cf - x2 > 0.0):
        raise ValueError("Invalid: (Cf - x) must be > 0 for both gears.")

    inva0 = _inv_angle(alpha0)
    invaw = inva0 + (2.0 * (x1 + x2) / _safe(z1 + z2, 1e-12)) * math.tan(alpha0)
    alpha_w = _inv_to_angle(invaw)

    a12 = ((z1 + z2) * m * 0.5) * (math.cos(alpha0) / max(1e-12, math.cos(alpha_w)))

    rg1 = ro1 * math.cos(alpha0)
    rg2 = ro2 * math.cos(alpha0)
    rw1 = rg1 / max(1e-12, math.cos(alpha_w))
    rw2 = rg2 / max(1e-12, math.cos(alpha_w))
    db1 = 2.0 * rw1
    db2 = 2.0 * rw2

    d01 = m * z1
    d02 = m * z2
    dg1 = d01 * math.cos(alpha0)
    dg2 = d02 * math.cos(alpha0)
    dk1 = 2.0 * ra1
    dk2 = 2.0 * ra2
    df1 = 2.0 * rf1
    df2 = 2.0 * rf2

    term1 = ra1 * ra1 - rb1 * rb1
    term2 = ra2 * ra2 - rb2 * rb2
    if term1 <= 0.0 or term2 <= 0.0:
        raise ValueError("Invalid: ra <= rb for one gear (no involute contact).")

    l_oa = math.sqrt(term1) + math.sqrt(term2) - a12 * math.sin(alpha_w)
    if not (l_oa > 0.0):
        raise ValueError("Invalid LOA length (<=0).")

    pb = math.pi * m * math.cos(alpha0)
    eps = l_oa / _safe(pb, 1e-12)

    s_a = math.sqrt(term2) - rw2 * math.sin(alpha_w)
    s_r = math.sqrt(term1) - rw1 * math.sin(alpha_w)
    n_pts = max(50, int(round(n_lo)))
    s_arr = _linspace(-s_a, s_r, n_pts)
    c = math.cos(alpha_w)
    s = math.sin(alpha_w)
    x_loa = (s_arr * c).tolist()
    y_loa = (rw1 + s_arr * s).tolist()

    return {
        "ro1": ro1,
        "ro2": ro2,
        "rb1": rb1,
        "rb2": rb2,
        "ra1": ra1,
        "ra2": ra2,
        "rf1": rf1,
        "rf2": rf2,
        "inva0": inva0,
        "invaw": invaw,
        "alphaW": alpha_w,
        "a12": a12,
        "rg1": rg1,
        "rg2": rg2,
        "rw1": rw1,
        "rw2": rw2,
        "d01": d01,
        "d02": d02,
        "dg1": dg1,
        "dg2": dg2,
        "db1": db1,
        "db2": db2,
        "dk1": dk1,
        "dk2": dk2,
        "df1": df1,
        "df2": df2,
        "pb": pb,
        "L": l_oa,
        "eps": eps,
        "xLOA": x_loa,
        "yLOA": y_loa,
    }


def compute_involute_loa_and_profile(
    m: float,
    z1: int,
    z2: int,
    alpha_rad: float,
    ck: float,
    cf: float,
    x1: float,
    x2: float,
    n: int,
) -> dict[str, Any]:
    if not (m > 0):
        raise ValueError("Module m must be > 0.")
    if not (z1 >= 3 and float(z1).is_integer()):
        raise ValueError("z1 must be integer >= 3.")
    if not (z2 >= 3 and float(z2).is_integer()):
        raise ValueError("z2 must be integer >= 3.")
    if not (0 < alpha_rad < math.pi / 2):
        raise ValueError("Pressure angle must be in (0, 90°).")
    if not (ck > 0 and cf > 0):
        raise ValueError("Ck and Cf must be > 0.")
    if not (n >= 200 and math.isfinite(n)):
        raise ValueError("Sampling points N must be >= 200.")

    ro1 = 0.5 * z1 * m
    ro2 = 0.5 * z2 * m
    rb1 = ro1 * math.cos(alpha_rad)
    rb2 = ro2 * math.cos(alpha_rad)

    ra1 = ro1 + (ck + x1) * m
    ra2 = ro2 + (ck + x2) * m
    rf1 = ro1 - (cf - x1) * m
    rf2 = ro2 - (cf - x2) * m

    yr = _linspace(-cf * m, ck * m, int(n))
    xr = -math.tan(alpha_rad) * yr
    dydx = np.full_like(yr, -1.0 / max(1e-12, math.tan(alpha_rad)))

    k_arr = -(yr * dydx + xr)
    theta = k_arr / _safe(ro1, 1e-12)
    xcp = xr + k_arr
    ycp = yr

    r_a1 = np.hypot(xcp, ycp + ro1)
    r_a2 = np.hypot(xcp, ycp - ro2)
    mask = (r_a1 <= ra1) & (r_a2 <= ra2)

    runs: list[tuple[int, int]] = []
    in_run = False
    s_idx = 0
    for i, ok in enumerate(mask.tolist()):
        if not in_run and ok:
            in_run = True
            s_idx = i
        if in_run and ((not ok) or i == len(mask) - 1):
            e_idx = i if (ok and i == len(mask) - 1) else i - 1
            runs.append((s_idx, e_idx))
            in_run = False

    if not runs:
        raise ValueError("LOA extraction failed (no valid contact segment).")

    best = runs[0]
    best_len = -1.0
    for a, b in runs:
        dx = np.diff(xcp[a : b + 1])
        dy = np.diff(ycp[a : b + 1])
        length = float(np.sum(np.hypot(dx, dy)))
        if length > best_len:
            best_len = length
            best = (a, b)

    i0, i1 = best
    x_loa = xcp[i0 : i1 + 1]
    y_loa = ycp[i0 : i1 + 1]
    th_loa = theta[i0 : i1 + 1]
    k_loa = k_arr[i0 : i1 + 1]

    cos_t = np.cos(th_loa)
    sin_t = np.sin(th_loa)
    x_pp = x_loa * cos_t - (y_loa + ro1) * sin_t
    y_pp = x_loa * sin_t + (y_loa + ro1) * cos_t - ro1

    return {
        "m": m,
        "z1": z1,
        "z2": z2,
        "alphaRad": alpha_rad,
        "Ck": ck,
        "Cf": cf,
        "x1": x1,
        "x2": x2,
        "ro1": ro1,
        "ro2": ro2,
        "rb1": rb1,
        "rb2": rb2,
        "ra1": ra1,
        "ra2": ra2,
        "rf1": rf1,
        "rf2": rf2,
        "xLOA": x_loa.tolist(),
        "yLOA": y_loa.tolist(),
        "thLOA": th_loa.tolist(),
        "KLOA": k_loa.tolist(),
        "xrLOA": (x_loa - k_loa).tolist(),
        "yrLOA": y_loa.tolist(),
        "xPP": x_pp.tolist(),
        "yPP": y_pp.tolist(),
    }


def check_operating_interference(oper: dict[str, Any]) -> dict[str, Any]:
    x_loa = np.asarray(oper["xLOA"], dtype=float)
    y_loa = np.asarray(oper["yLOA"], dtype=float)
    a12 = float(oper["a12"])
    rb1 = float(oper["rb1"])
    rb2 = float(oper["rb2"])
    ro1 = float(oper["ro1"])
    ro2 = float(oper["ro2"])

    tol = 1e-6 * max(ro1, ro2)
    r1 = np.hypot(x_loa, y_loa)
    r2 = np.hypot(x_loa, y_loa - a12)
    min_r1 = float(np.min(r1))
    min_r2 = float(np.min(r2))
    margin1 = min_r1 - rb1
    margin2 = min_r2 - rb2
    return {
        "minR1": min_r1,
        "minR2": min_r2,
        "margin1": margin1,
        "margin2": margin2,
        "pinionInterference": bool(margin1 < -tol),
        "gearInterference": bool(margin2 < -tol),
    }


def _x_max_eq21(z: int) -> float:
    if 6 <= z <= 10:
        return 0.60
    if 10 < z <= 50:
        return 0.50 - 0.01 * z
    if z > 50:
        return 1.00
    return float("nan")


def _x_max_eq22(z: int) -> float:
    if 6 <= z <= 12:
        return 0.05 * (18 - z)
    if 12 < z <= 20:
        return 0.0375 * (20 - z)
    if 20 < z <= 50:
        return (z - 20) / 60
    if z > 50:
        return 0.50
    return float("nan")


def check_geometry_constraints(z: int, alpha_rad: float, cf: float, cc: float, x: float, cs: float) -> dict[str, Any]:
    sin_a = math.sin(alpha_rad)
    tan_a = math.tan(alpha_rad)
    cos_a = math.cos(alpha_rad)
    den20 = max(1e-12, sin_a * sin_a)

    z_min20 = 2.0 * (1.0 - x) / den20
    z_ok = z >= (z_min20 - 1e-9)

    x_max21 = _x_max_eq21(int(z))
    x_max22 = _x_max_eq22(int(z))
    if math.isfinite(x_max21) and math.isfinite(x_max22):
        x_max = min(x_max21, x_max22)
    elif math.isfinite(x_max21):
        x_max = x_max21
    else:
        x_max = x_max22
    x_ok = (x <= (x_max + 1e-9)) if math.isfinite(x_max) else True

    cf_max23 = ((1.0 - cs) * math.pi) / (2.0 * max(1e-12, tan_a))
    cf_ok = cf <= (cf_max23 + 1e-9)

    den24 = cos_a - (1.0 - sin_a) * tan_a
    cc_max24 = (((1.0 - cs) * math.pi * 0.5) - cf * tan_a) / max(1e-12, den24)
    cc_ok = cc <= (cc_max24 + 1e-9)

    return {
        "zMin20": z_min20,
        "zOk": bool(z_ok),
        "xMax21": x_max21,
        "xMax22": x_max22,
        "xMax": x_max,
        "xOk": bool(x_ok),
        "CfMax23": cf_max23,
        "cfOk": bool(cf_ok),
        "CcMax24": cc_max24,
        "ccOk": bool(cc_ok),
        "hasUndercut": bool(not z_ok),
        "cs": cs,
        "Cc": cc,
        "Cf": cf,
        "x": x,
    }


def compute_geometry_outputs(payload: dict[str, Any]) -> dict[str, Any]:
    m = float(payload["m"])
    z1 = int(payload["z1"])
    z2 = int(payload["z2"])
    alpha_rad = float(payload["alpha_rad"])
    ck = float(payload["Ck"])
    cf = float(payload["Cf"])
    x1 = float(payload["x1"])
    x2 = float(payload["x2"])
    cc = float(payload["Cc"])
    cs1 = float(payload.get("cs1", 0.49))
    cs2 = float(payload.get("cs2", 0.49))
    n = int(payload.get("N", 1200))

    oper = compute_operating_from_shifts(
        m=m,
        z1=z1,
        z2=z2,
        alpha0=alpha_rad,
        ck=ck,
        cf=cf,
        x1=x1,
        x2=x2,
        n_lo=max(200, min(1200, n)),
    )
    base = compute_involute_loa_and_profile(
        m=m,
        z1=z1,
        z2=z2,
        alpha_rad=alpha_rad,
        ck=ck,
        cf=cf,
        x1=x1,
        x2=x2,
        n=max(200, n),
    )
    inter = check_operating_interference(oper)
    uc1 = check_geometry_constraints(z=z1, alpha_rad=alpha_rad, cf=cf, cc=cc, x=x1, cs=cs1)
    uc2 = check_geometry_constraints(z=z2, alpha_rad=alpha_rad, cf=cf, cc=cc, x=x2, cs=cs2)

    return {
        "base": base,
        "oper": oper,
        "qc": {
            "inter": inter,
            "uc1": uc1,
            "uc2": uc2,
        },
    }
