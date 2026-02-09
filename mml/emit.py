from .interact import resolve_params
from .vision import normalize_vision


def _scale_point(p, scale):
    return [round(float(p[0]) * scale, 3), round(float(p[1]) * scale, 3)]


def emit_mml(vision, params, image_path, prompt_fn=None, include_intent=False, inferred_part=None):
    vision = normalize_vision(vision)
    resolved, chat = resolve_params(
        vision,
        params,
        prompt_fn=prompt_fn,
        include_intent=include_intent,
        inferred_part=inferred_part,
    )
    px_to_mm = resolved["px_to_mm"]

    outline = vision.get("outline", {}) or {}
    outline_px = outline.get("points_px", []) or []
    outline_type = outline.get("type") or "polygon"
    outline_mm = []
    for p in outline_px:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            continue
        outline_mm.append(_scale_point(p, px_to_mm))

    holes_mm = []
    hole_diameters = resolved.get("hole_diameters_mm") or []
    for idx, h in enumerate(vision.get("holes", []) or []):
        if not isinstance(h, dict):
            continue
        center_px = h.get("center_px")
        if not center_px or len(center_px) != 2:
            continue
        center_mm = _scale_point(center_px, px_to_mm)
        diameter_mm = None
        if idx < len(hole_diameters):
            diameter_mm = round(float(hole_diameters[idx]), 3)
        holes_mm.append({"center_mm": center_mm, "diameter_mm": diameter_mm})

    hole_standard = resolved["hole_standard"] or "custom"
    if resolved["hole_diameter_mm"] is not None and hole_standard == "custom":
        hole_standard = "custom"

    mml = {
        "part": params.get("part_name") or "Unknown",
        "units": "mm",
        "scale": {"px_to_mm": px_to_mm},
        "material": {"name": params.get("material") or "A5052"},
        "process": {"name": params.get("process") or "sheet_metal"},
        "geometry": {
            "outline": {"type": outline_type, "points_mm": outline_mm},
            "holes": [
                {
                    "type": "clearance",
                    "standard": hole_standard,
                    "diameter_mm": h["diameter_mm"],
                    "center_mm": h["center_mm"],
                }
                for h in holes_mm
            ],
        },
        "constraints": [],
        "provenance": {
            "vision": {"file": image_path, "version": "0.1"},
            "chat": chat,
        },
    }
    intent = resolved.get("intent")
    if include_intent:
        mml["intent"] = intent or {}

    bend_lines = vision.get("bend_lines", []) or []
    if bend_lines and isinstance(bend_lines[0], dict):
        line_px = bend_lines[0].get("line_px") or []
        if len(line_px) == 2 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in line_px):
            line_mm = [_scale_point(p, px_to_mm) for p in line_px]
        else:
            line_mm = None
    else:
        line_mm = None
    if line_mm:
        mml["geometry"]["bend"] = {
            "line_mm": line_mm,
            "angle_deg": resolved["bend_angle_deg"] or 90.0,
            "inner_radius_mm": resolved["bend_radius_mm"] or 1.0,
        }

    if resolved["thickness_mm"] is not None:
        mml["constraints"].append({"kind": "min_thickness", "value_mm": resolved["thickness_mm"]})
    mml["constraints"].append({"kind": "bend_radius_gte_thickness"})
    mml["constraints"].append({"kind": "edge_distance_gte", "multiplier": 2.0})

    hole_conf = [h.get("confidence", 0) for h in vision.get("holes", [])]
    bend_conf = [b.get("confidence", 0) for b in vision.get("bend_lines", [])]
    report = {
        "scale_px_to_mm": px_to_mm,
        "hole_standard": hole_standard,
        "hole_diameters_mm": [h["diameter_mm"] for h in holes_mm],
        "questions": chat["questions"],
        "answers": chat["answers"],
        "vision_confidence": {
            "holes_avg": round(sum(hole_conf) / len(hole_conf), 3) if hole_conf else None,
            "bend_lines_avg": round(sum(bend_conf) / len(bend_conf), 3) if bend_conf else None,
        },
        "decisions": [],
        "notes": [],
    }
    if params.get("plate_width_mm") is None and params.get("px_to_mm") is None:
        report["notes"].append("Scale not provided; px_to_mm assumed as 1.0")
    if resolved.get("unify_holes") is True:
        report["decisions"].append("hole_size_normalized")
    if any(d is None for d in report["hole_diameters_mm"]):
        report["notes"].append("Hole diameter unresolved")

    return mml, report
