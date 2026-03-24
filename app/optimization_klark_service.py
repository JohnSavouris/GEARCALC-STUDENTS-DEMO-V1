from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.efficiency_service import compute_efficiency_outputs
from app.geometry_service import (
    check_geometry_constraints,
    check_operating_interference,
    compute_operating_from_shifts,
)


STD_MODULE_SERIES = np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0], dtype=float)


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _safe(v: float, lo: float) -> float:
    return float(max(lo, v))


@dataclass
class BaseState:
    z1: int
    z2: int
    m: float
    b: float
    x1: float
    x2: float
    alpha_deg: float
    torque_nm: float
    n1_rpm: float
    ck: float
    cf: float
    cc: float
    cs1: float
    cs2: float
    sf_min: float
    sh_min: float
    eff_pct: float
    loss_w: float
    eps: float
    mass_kg: float
    ste_ptp: float


def _xup_xlow_professor_code(z: int) -> tuple[float, float]:
    # This follows the exact logic in optimization_codes/pymoo_problem.py.
    if z <= 10:
        xup = 0.60
    elif z <= 50:
        xup = 0.50 + 0.01 * z
    else:
        xup = 1.0

    if z <= 12:
        xlow = 0.05 * (18 - z)
    elif z <= 20:
        xlow = 0.0375 * (20 - z)
    elif z <= 50:
        xlow = (20 - z) / 60
    else:
        xlow = -0.50
    return float(xup), float(xlow)


def _estimate_mass_kg(ro1: float, ro2: float, ra1: float, ra2: float, rf1: float, rf2: float, z1: int, z2: int, m: float, b: float, rho: float) -> float:
    def one(ra: float, rf: float, ro: float, z: int) -> float:
        r_tip = max(float(ra), 1.0)
        r_pitch = max(float(ro), 1.0)
        r_root = max(0.60 * r_tip, float(rf))
        rim_inner = max(5.0, r_root - 0.9 * m)
        bore = max(8.0, 0.36 * r_pitch)
        web_outer = max(bore + 2.0, 0.70 * rim_inner)
        ring_area = np.pi * max(0.0, r_tip * r_tip - rim_inner * rim_inner)
        web_area = np.pi * max(0.0, web_outer * web_outer - bore * bore)
        tooth_fill = 0.58 - 0.06 * min(1.0, max(0.0, (z - 12) / 120))
        web_fill = 0.22
        area_eff = ring_area * tooth_fill + web_area * web_fill
        return area_eff * b

    v1 = one(ra1, rf1, ro1, z1)
    v2 = one(ra2, rf2, ro2, z2)
    vol_m3 = (v1 + v2) * 1e-9
    return float(max(1e-6, rho * vol_m3))


def _predict_safeties(base: BaseState, z1: int, z2: int, m: float, b: float, x1: float, x2: float, alpha_deg: float) -> tuple[float, float]:
    rb = b / _safe(base.b, 1e-9)
    rm = m / _safe(base.m, 1e-9)
    rz = z1 / _safe(base.z1, 1e-9)
    rx1 = x1 - base.x1
    rx2 = x2 - base.x2
    ra = _safe(base.alpha_deg, 1e-9) / _safe(alpha_deg, 1e-9)

    sf = base.sf_min
    sh = base.sh_min
    sf *= rb**1.0
    sf *= rm**1.65
    sf *= rz**0.60
    sf *= ra**0.15
    sf *= 1.0 + 0.12 * max(0.0, rx1) + 0.05 * max(0.0, rx2)

    sh *= rb**0.50
    sh *= rm**1.05
    sh *= rz**0.40
    sh *= ra**0.12
    sh *= 1.0 + 0.08 * max(0.0, rx1 + rx2)
    return float(sf), float(sh)


def _predict_ste_ptp(base: BaseState, m: float, b: float, eps: float, x1: float, x2: float, alpha_deg: float) -> float:
    v = base.ste_ptp
    v *= (_safe(base.m, 1e-9) / _safe(m, 1e-9)) ** 1.15
    v *= (_safe(base.b, 1e-9) / _safe(b, 1e-9)) ** 0.72
    v *= (_safe(base.eps, 1e-9) / _safe(eps, 1e-9)) ** 0.55
    v *= (_safe(alpha_deg, 1e-9) / _safe(base.alpha_deg, 1e-9)) ** 0.28
    v *= 1.0 + 0.06 * abs(x1 - x2)
    return float(_clamp(v, 1e-6, 1e4))


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _pareto_indices(f: np.ndarray) -> list[int]:
    n = int(f.shape[0])
    keep: list[int] = []
    for i in range(n):
        dom = False
        for j in range(n):
            if i == j:
                continue
            if _dominates(f[j], f[i]):
                dom = True
                break
        if not dom:
            keep.append(i)
    return keep


def _constraint_penalty(g: list[float]) -> float:
    scales = np.array([0.2, 0.5, 0.5, 5.0, 10.0, 10.0, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 5.0, 5.0], dtype=float)
    gg = np.array(g, dtype=float)
    v = np.maximum(gg, 0.0) / scales
    return float(np.sum(v))


def _diagnosis(sf: float, sh: float, eps: float, feasible_ratio: float) -> dict[str, Any]:
    tags: list[str] = []
    if sf < 1.0:
        tags.append("Bending constraint active")
    if sh < 1.0:
        tags.append("Pitting/contact constraint active")
    if eps < 1.2:
        tags.append("Contact ratio close to limit")
    if feasible_ratio < 0.2:
        tags.append("Narrow feasible domain")
    if not tags:
        tags = ["Balanced design region"]

    sev = "low"
    if sf < 0.9 or sh < 0.9 or eps < 1.15:
        sev = "high"
    elif sf < 1.05 or sh < 1.05 or feasible_ratio < 0.45:
        sev = "medium"
    return {"severity": sev, "dominant_modes": tags}


def optimize_design_klark(payload: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(payload.get("seed", 7)))

    z1_0 = int(round(float(payload["z1"])))
    z2_0 = int(round(float(payload["z2"])))
    ratio = float(payload.get("ratio", z2_0 / _safe(z1_0, 1e-9)))

    m0 = float(payload["m"])
    b0 = float(payload["b_mm"])
    x1_0 = float(payload.get("x1", 0.0))
    x2_0 = float(payload.get("x2", 0.0))
    alpha0 = float(payload.get("alpha_deg", 20.0))

    ck = float(payload.get("Ck", 1.0))
    cf = float(payload.get("Cf", 1.25))
    cc = float(payload.get("Cc", 0.25))
    cs1 = float(payload.get("cs1", 0.49))
    cs2 = float(payload.get("cs2", 0.49))

    torque_nm = float(payload.get("torque_nm", 30.0))
    n1_rpm = float(payload.get("n1_rpm", 1500.0))
    rho = float(payload.get("density_kg_m3", 7850.0))

    include_shifts = bool(payload.get("include_shifts", False))
    include_alpha = bool(payload.get("include_alpha", False))
    std_module_only = bool(payload.get("std_module_only", True))

    z1_min = int(round(float(payload.get("z1_min", max(6, z1_0 - 8)))))
    z1_max = int(round(float(payload.get("z1_max", z1_0 + 12))))
    m_min = float(payload.get("m_min", 1.0))
    m_max = float(payload.get("m_max", max(3.0, m0)))
    b_min = float(payload.get("b_min", max(6.0, 0.7 * b0)))
    b_max = float(payload.get("b_max", min(180.0, 1.5 * b0 + 10.0)))
    x1_min = float(payload.get("x1_min", 0.0))
    x1_max = float(payload.get("x1_max", 1.0))
    x2_min = float(payload.get("x2_min", -0.5))
    x2_max = float(payload.get("x2_max", 1.0))
    a_min = float(payload.get("alpha_min", 20.0))
    a_max = float(payload.get("alpha_max", 30.0))
    if z1_min > z1_max:
        z1_min, z1_max = z1_max, z1_min
    if m_min > m_max:
        m_min, m_max = m_max, m_min
    if b_min > b_max:
        b_min, b_max = b_max, b_min
    if x1_min > x1_max:
        x1_min, x1_max = x1_max, x1_min
    if x2_min > x2_max:
        x2_min, x2_max = x2_max, x2_min
    if a_min > a_max:
        a_min, a_max = a_max, a_min

    w_mass = float(payload.get("w_mass", 0.35))
    w_loss = float(payload.get("w_loss", 0.30))
    w_ste = float(payload.get("w_ste", 0.25))
    w_inv_eps = float(payload.get("w_inv_eps", 0.10))
    w_sum = _safe(w_mass + w_loss + w_ste + w_inv_eps, 1e-9)
    w_mass, w_loss, w_ste, w_inv_eps = w_mass / w_sum, w_loss / w_sum, w_ste / w_sum, w_inv_eps / w_sum

    top_k = int(payload.get("top_k", 14))
    n_samples = int(payload.get("n_samples", 420))
    n_samples = int(np.clip(n_samples, 120, 3000))
    base_ste_ptp = float(payload.get("base_ste_ptp", 8.0))
    sf_base = float(payload.get("sf_min", 1.1))
    sh_base = float(payload.get("sh_min", 1.05))
    eps_base = float(payload.get("contact_ratio", 1.5))
    eff_base = float(payload.get("efficiency_percent", 98.8))
    loss_base = float(payload.get("power_loss_w", 5.0))
    mass_base = float(payload.get("mass_kg", 1.0))

    base = BaseState(
        z1=z1_0,
        z2=z2_0,
        m=m0,
        b=b0,
        x1=x1_0,
        x2=x2_0,
        alpha_deg=alpha0,
        torque_nm=torque_nm,
        n1_rpm=n1_rpm,
        ck=ck,
        cf=cf,
        cc=cc,
        cs1=cs1,
        cs2=cs2,
        sf_min=sf_base,
        sh_min=sh_base,
        eff_pct=eff_base,
        loss_w=loss_base,
        eps=eps_base,
        mass_kg=mass_base,
        ste_ptp=base_ste_ptp,
    )

    def sample_m() -> float:
        if not std_module_only:
            return float(rng.uniform(m_min, m_max))
        valid = STD_MODULE_SERIES[(STD_MODULE_SERIES >= m_min - 1e-12) & (STD_MODULE_SERIES <= m_max + 1e-12)]
        if valid.size == 0:
            return float(np.clip(m0, m_min, m_max))
        return float(valid[rng.integers(0, valid.size)])

    # baseline + random design set
    candidates: list[dict[str, Any]] = []
    design_vectors: list[tuple[float, ...]] = [(float(z1_0), m0, b0, x1_0, x2_0, alpha0)]
    for _ in range(n_samples - 1):
        z1 = int(rng.integers(z1_min, z1_max + 1))
        m = sample_m()
        b = float(rng.uniform(b_min, b_max))
        if include_shifts:
            x1 = float(rng.uniform(x1_min, x1_max))
            x2 = float(rng.uniform(x2_min, x2_max))
        else:
            x1, x2 = x1_0, x2_0
        if include_alpha:
            alpha = float(rng.uniform(a_min, a_max))
        else:
            alpha = alpha0
        design_vectors.append((float(z1), m, b, x1, x2, alpha))

    for i, (z1f, m, b, x1, x2, alpha) in enumerate(design_vectors):
        z1 = int(round(z1f))
        z2 = int(round(_safe(ratio, 1.01) * z1))
        alpha = _clamp(alpha, 15.0, 35.0)
        m = _clamp(m, 0.8, 12.0)
        b = _clamp(b, 4.0, 220.0)
        x1 = _clamp(x1, -2.0, 2.0)
        x2 = _clamp(x2, -2.0, 2.0)
        try:
            oper = compute_operating_from_shifts(
                m=m,
                z1=z1,
                z2=z2,
                alpha0=np.deg2rad(alpha),
                ck=ck,
                cf=cf,
                x1=x1,
                x2=x2,
                n_lo=600,
            )
            inter = check_operating_interference(oper)
            uc1 = check_geometry_constraints(z1, np.deg2rad(alpha), cf, cc, x1, cs1)
            uc2 = check_geometry_constraints(z2, np.deg2rad(alpha), cf, cc, x2, cs2)
            eps = float(oper["eps"])

            sf, sh = _predict_safeties(base, z1, z2, m, b, x1, x2, alpha)
            ste_ptp = _predict_ste_ptp(base, m, b, eps, x1, x2, alpha)

            eff_payload = {
                "z1": z1,
                "z2": z2,
                "m": m,
                "alpha_deg": alpha,
                "x1": x1,
                "x2": x2,
                "ck": ck,
                "b_mm": b,
                "n1_rpm": n1_rpm,
                "torque_nm": torque_nm,
                "eps_alpha": eps,
                "lube_family": str(payload.get("lube_family", "mineral")),
                "iso_vg": float(payload.get("iso_vg", 68.0)),
                "oil_temp_c": float(payload.get("oil_temp_c", 60.0)),
                "additive": str(payload.get("additive", "none")),
                "alpha_pv": float(payload.get("alpha_pv", 2e-8)),
                "mu_lim": float(payload.get("mu_lim", 0.11)),
                "e_mpa": float(payload.get("e_mpa", 206000.0)),
                "nu": float(payload.get("nu", 0.30)),
                "roughness_um": float(payload.get("roughness_um", 0.30)),
                "n_points": 450,
            }
            eff_out = compute_efficiency_outputs(eff_payload)
            loss_w = float(eff_out["scalars"]["P_loss_w"])
            eff_pct = float(eff_out["scalars"]["efficiency_percent"])
            mu_mean = float(eff_out["scalars"]["mu_mean"])
            lam_mean = float(eff_out["scalars"]["lambda_mean"])
            mass = _estimate_mass_kg(
                ro1=float(oper["ro1"]),
                ro2=float(oper["ro2"]),
                ra1=float(oper["ra1"]),
                ra2=float(oper["ra2"]),
                rf1=float(oper["rf1"]),
                rf2=float(oper["rf2"]),
                z1=z1,
                z2=z2,
                m=m,
                b=b,
                rho=rho,
            )

            # Constraints from professor script
            xup1, xlow1 = _xup_xlow_professor_code(z1)
            xup2, xlow2 = _xup_xlow_professor_code(z2)
            tan_a = np.tan(np.deg2rad(alpha))
            cf1max = (1.0 - cs1) * np.pi / 2.0 / max(1e-12, tan_a)
            cf2max = (1.0 - cs2) * np.pi / 2.0 / max(1e-12, tan_a)
            den_cc = (np.cos(np.deg2rad(alpha)) - (1.0 - np.sin(np.deg2rad(alpha))) * tan_a)
            cc1max = (((1.0 - cs1) * np.pi / 2.0) - cf * tan_a) / max(1e-12, den_cc)
            cc2max = (((1.0 - cs2) * np.pi / 2.0) - cf * tan_a) / max(1e-12, den_cc)

            b_to_m = b / _safe(m, 1e-12)
            g = [
                1.2 - eps,  # g1
                1.0 - sh,  # g2
                1.0 - sf,  # g3
                1.0 - b_to_m,  # g4
                -10.0,  # g5 disabled as in professor script
                -10.0,  # g6 disabled as in professor script
                xlow1 - x1,  # g7
                x1 - xup1,  # g8
                xlow2 - x2,  # g9
                x2 - xup2,  # g10
                cf - cf1max,  # g11
                cf - cf2max,  # g12
                cc - cc1max,  # g13
                cc - cc2max,  # g14
                b_to_m - 40.0,  # g15
                -b_to_m + 20.0,  # g16
            ]
            penalty = _constraint_penalty(g)
            feasible = bool(all(gg <= 1e-9 for gg in g))
            undercut_ok = bool(uc1["zOk"] and uc2["zOk"])
            interference_ok = bool(not inter["pinionInterference"] and not inter["gearInterference"])
            actions = []
            if z1 != z1_0:
                actions.append(f"z1: {z1_0} -> {z1}")
            if abs(m - m0) > 1e-9:
                actions.append(f"m: {m0:.2f} -> {m:.2f}")
            if abs(b - b0) > 1e-9:
                actions.append(f"b: {b0:.1f} -> {b:.1f}")
            if include_shifts:
                if abs(x1 - x1_0) > 1e-9:
                    actions.append(f"x1: {x1_0:.3f} -> {x1:.3f}")
                if abs(x2 - x2_0) > 1e-9:
                    actions.append(f"x2: {x2_0:.3f} -> {x2:.3f}")
            if include_alpha and abs(alpha - alpha0) > 1e-9:
                actions.append(f"α0: {alpha0:.2f} -> {alpha:.2f}")

            candidates.append(
                {
                    "id": f"K{i+1:04d}",
                    "idx": i,
                    "vars": {"z1": z1, "z2": z2, "m": m, "b_mm": b, "x1": x1, "x2": x2, "alpha_deg": alpha},
                    "objectives": {"mass_kg": mass, "loss_w": loss_w, "ste_ptp": ste_ptp, "inv_eps": 1.0 / _safe(eps, 1e-12)},
                    "predicted": {
                        "sf_min": sf,
                        "sh_min": sh,
                        "efficiency_percent": eff_pct,
                        "contact_ratio": eps,
                        "mass_kg": mass,
                        "power_loss_w": loss_w,
                        "ste_ptp": ste_ptp,
                        "mu_mean": mu_mean,
                        "lambda_mean": lam_mean,
                        "dynamic_index": max(0.02, ste_ptp / _safe(base_ste_ptp, 1e-9)),
                    },
                    "constraints": {"g": g, "penalty": penalty, "feasible": feasible, "undercut_ok": undercut_ok, "interference_ok": interference_ok},
                    "actions": actions,
                    "title": " / ".join(actions) if actions else "Baseline geometry",
                    "note": "Kaligeros-adapted multi-objective candidate",
                }
            )
        except Exception:
            continue

    if not candidates:
        return {"ok": False, "detail": "No valid optimization candidates were generated."}

    f_mass = np.array([c["objectives"]["mass_kg"] for c in candidates], dtype=float)
    f_loss = np.array([c["objectives"]["loss_w"] for c in candidates], dtype=float)
    f_ste = np.array([c["objectives"]["ste_ptp"] for c in candidates], dtype=float)
    f_inv_eps = np.array([c["objectives"]["inv_eps"] for c in candidates], dtype=float)
    pen = np.array([c["constraints"]["penalty"] for c in candidates], dtype=float)
    feas = np.array([bool(c["constraints"]["feasible"]) for c in candidates], dtype=bool)

    def norm(v: np.ndarray) -> np.ndarray:
        lo = float(np.min(v))
        hi = float(np.max(v))
        if hi - lo < 1e-12:
            return np.zeros_like(v)
        return (v - lo) / (hi - lo)

    n_mass = norm(f_mass)
    n_loss = norm(f_loss)
    n_ste = norm(f_ste)
    n_inv_eps = norm(f_inv_eps)
    weighted_cost = w_mass * n_mass + w_loss * n_loss + w_ste * n_ste + w_inv_eps * n_inv_eps
    score = 100.0 * (1.0 - weighted_cost) - 20.0 * pen

    for i, c in enumerate(candidates):
        c["score"] = float(score[i])

    # Baseline candidate: first matching exact baseline, else nearest by weighted cost.
    baseline_idx = 0
    for i, c in enumerate(candidates):
        vv = c["vars"]
        if (
            int(vv["z1"]) == z1_0
            and abs(vv["m"] - m0) < 1e-9
            and abs(vv["b_mm"] - b0) < 1e-9
            and abs(vv["x1"] - x1_0) < 1e-9
            and abs(vv["x2"] - x2_0) < 1e-9
            and abs(vv["alpha_deg"] - alpha0) < 1e-9
        ):
            baseline_idx = i
            break

    # Pareto among feasible in 3 objectives (mass, losses, STE).
    f3 = np.column_stack([f_mass, f_loss, f_ste])
    feasible_idx = np.where(feas)[0].tolist()
    if feasible_idx:
        pf_local = _pareto_indices(f3[feasible_idx, :])
        pf_idx = [feasible_idx[k] for k in pf_local]
    else:
        pf_idx = []

    # Scenario ranking
    order = np.argsort(-score)
    ranked = [candidates[int(i)] for i in order]
    top = ranked[: max(5, int(top_k))]
    best = top[0]
    base_score = float(score[baseline_idx])
    for s in top:
        idx = int(s["idx"])
        s["score_gain"] = float(s["score"] - base_score)
        s["points"] = {
            "mass_penalty": float(n_mass[idx] * 30.0),
            "loss_penalty": float(n_loss[idx] * 30.0),
            "ste_penalty": float(n_ste[idx] * 25.0),
            "eps_penalty": float(n_inv_eps[idx] * 15.0),
            "constraint_penalty": float(pen[idx] * 20.0),
            "total": float(s["score"]),
        }
        s["params"] = {
            "z1": float(s["vars"]["z1"]),
            "z2": float(s["vars"]["z2"]),
            "m": float(s["vars"]["m"]),
            "b_mm": float(s["vars"]["b_mm"]),
            "x1": float(s["vars"]["x1"]),
            "x2": float(s["vars"]["x2"]),
            "alpha_deg": float(s["vars"]["alpha_deg"]),
            "torque_nm": float(torque_nm),
            "n1_rpm": float(n1_rpm),
        }

    # Sensitivity around baseline
    sensitivity: list[dict[str, Any]] = []
    var_keys = [("z1", "z1"), ("m", "module m"), ("b_mm", "face width b")]
    if include_shifts:
        var_keys.extend([("x1", "profile shift x1"), ("x2", "profile shift x2")])
    if include_alpha:
        var_keys.append(("alpha_deg", "pressure angle alpha0"))
    base_vars = candidates[baseline_idx]["vars"]
    for key, label in var_keys:
        vals = np.array([c["vars"][key] for c in candidates], dtype=float)
        if np.ptp(vals) < 1e-9:
            continue
        corr = np.corrcoef(vals, score)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
        sensitivity.append({"lever": label, "gain_points": float(28.0 * corr), "gain_percent": float(100.0 * corr)})
    sensitivity.sort(key=lambda x: abs(float(x["gain_points"])), reverse=True)

    feasible_ratio = float(np.mean(feas))
    diag = _diagnosis(
        sf=float(candidates[baseline_idx]["predicted"]["sf_min"]),
        sh=float(candidates[baseline_idx]["predicted"]["sh_min"]),
        eps=float(candidates[baseline_idx]["predicted"]["contact_ratio"]),
        feasible_ratio=feasible_ratio,
    )

    # Advanced chart payloads (correlation matrix)
    corr_labels = ["z1", "m", "b", "x1", "x2", "alpha", "mass", "loss", "ste", "eps", "score"]
    mat = np.column_stack(
        [
            np.array([c["vars"]["z1"] for c in candidates], dtype=float),
            np.array([c["vars"]["m"] for c in candidates], dtype=float),
            np.array([c["vars"]["b_mm"] for c in candidates], dtype=float),
            np.array([c["vars"]["x1"] for c in candidates], dtype=float),
            np.array([c["vars"]["x2"] for c in candidates], dtype=float),
            np.array([c["vars"]["alpha_deg"] for c in candidates], dtype=float),
            f_mass,
            f_loss,
            f_ste,
            np.array([c["predicted"]["contact_ratio"] for c in candidates], dtype=float),
            score,
        ]
    )
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    pareto = [
        {
            "id": candidates[i]["id"],
            "mass_kg": float(f_mass[i]),
            "loss_w": float(f_loss[i]),
            "ste_ptp": float(f_ste[i]),
            "score": float(score[i]),
        }
        for i in pf_idx
    ]

    advice = [
        "The optimizer follows Kaligeros-style objectives: minimize mass, losses and STE amplitude while respecting constraints.",
        f"Feasible ratio in sampled set: {100*feasible_ratio:.1f}%.",
        f'Best candidate: {best["title"]} with score {best["score"]:.2f} (gain {best["score"]-base_score:+.2f}).',
    ]
    if not include_shifts:
        advice.append("Profile shifts were fixed (x1/x2 locked). Enable them to enlarge the feasible domain.")
    if not include_alpha:
        advice.append("Pressure angle alpha0 was fixed. Enable alpha optimization only if manufacturing/tooling supports it.")

    baseline = candidates[baseline_idx]

    return {
        "ok": True,
        "meta": {
            "engine": "Kaligeros-adapted",
            "samples": len(candidates),
            "feasible_ratio": feasible_ratio,
            "include_shifts": include_shifts,
            "include_alpha": include_alpha,
            "std_module_only": std_module_only,
            "weights": {"mass": w_mass, "loss": w_loss, "ste": w_ste, "inv_eps": w_inv_eps},
        },
        "diagnosis": diag,
        "score_summary": {
            "current": float(base_score),
            "best_reachable": float(best["score"]),
            "gain": float(best["score"] - base_score),
            "only_improving": bool(any(float(s["score"] - base_score) > 1e-9 for s in top)),
        },
        "baseline": {
            "id": baseline["id"],
            "predicted": baseline["predicted"],
            "params": baseline["params"] if "params" in baseline else {
                "z1": float(base_vars["z1"]),
                "z2": float(candidates[baseline_idx]["vars"]["z2"]),
                "m": float(base_vars["m"]),
                "b_mm": float(base_vars["b_mm"]),
                "x1": float(base_vars["x1"]),
                "x2": float(base_vars["x2"]),
                "alpha_deg": float(base_vars["alpha_deg"]),
            },
            "points": {"total": float(base_score)},
            "constraints": baseline["constraints"],
        },
        "best": best,
        "scenarios": top,
        "sensitivity": sensitivity,
        "pareto": pareto,
        "advanced": {
            "corr_labels": corr_labels,
            "corr_matrix": corr.tolist(),
            "scatter_pool": [
                {
                    "id": c["id"],
                    "z1": float(c["vars"]["z1"]),
                    "m": float(c["vars"]["m"]),
                    "b_mm": float(c["vars"]["b_mm"]),
                    "x1": float(c["vars"]["x1"]),
                    "x2": float(c["vars"]["x2"]),
                    "alpha_deg": float(c["vars"]["alpha_deg"]),
                    "mass_kg": float(c["objectives"]["mass_kg"]),
                    "loss_w": float(c["objectives"]["loss_w"]),
                    "ste_ptp": float(c["objectives"]["ste_ptp"]),
                    "eps": float(c["predicted"]["contact_ratio"]),
                    "score": float(c["score"]),
                    "feasible": bool(c["constraints"]["feasible"]),
                }
                for c in ranked[: min(220, len(ranked))]
            ],
        },
        "advice": advice,
        "points_legend": {
            "sf": "safety tendency via surrogate scaling",
            "sh": "contact tendency via surrogate scaling",
            "tip": "tip reserve is indirectly handled through constraints and module/shift trends",
            "eps": "contact-ratio contribution",
            "dyn": "dynamic tendency tied to STE surrogate",
            "eff": "efficiency/loss contribution",
            "mass_penalty": "mass objective term",
            "geometry_penalty": "sum of positive constraint violations",
            "cost_penalty": "weighted objective combination",
        },
    }

