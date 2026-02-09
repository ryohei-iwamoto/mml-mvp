"""
意図推定モジュール。

部品ライブラリと連携し、AIによる部品選択に対応。
"""

from typing import Dict, Any, Optional, List


def _outline_stats(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    radii = [((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in points]
    if not radii:
        return None
    mean = sum(radii) / len(radii)
    var = sum((r - mean) ** 2 for r in radii) / len(radii)
    return {"center": (cx, cy), "mean_r": mean, "std_r": var ** 0.5}


def infer_part_from_vision(vision):
    hint = vision.get("part_hint")
    if hint:
        label = str(hint)
        if label.lower() == "gear":
            label = "Gear"
        if label.lower() == "motor":
            label = "Motor"
        if label.lower() == "robotarm":
            label = "RobotArm"
        return {
            "label": label,
            "confidence": float(vision.get("part_hint_confidence", 0.75)),
        }

    outline = vision.get("outline", {}).get("points_px", [])
    holes = vision.get("holes", [])
    bends = vision.get("bend_lines", [])

    stats = _outline_stats(outline)
    if stats:
        std_ratio = stats["std_r"] / max(stats["mean_r"], 1.0)
        if holes and (std_ratio > 0.12 or len(outline) >= 24):
            return {"label": "Gear", "confidence": 0.7}
        if std_ratio < 0.08 and holes:
            return {"label": "Plate", "confidence": 0.6}

    if bends:
        return {"label": "Bracket", "confidence": 0.7}
    if outline and len(outline) == 4 and holes:
        return {"label": "Plate", "confidence": 0.6}
    if outline:
        return {"label": "Plate", "confidence": 0.5}
    return {"label": "Unknown", "confidence": 0.2}


def _map_hint_to_library_id(hint: str) -> Optional[str]:
    """ビジョンのヒントをライブラリ部品IDに対応付ける。"""
    hint_lower = hint.lower()

    mapping = {
        "gear": "spur_gear",
        "spur gear": "spur_gear",
        "helical gear": "helical_gear",
        "bevel gear": "bevel_gear",
        "rack": "rack",
        "bracket": "bracket",
        "plate": "plate",
        "motor": "motor",
        "shaft": "shaft",
        "bearing": "bearing",
        "bolt": "bolt",
        "nut": "nut",
        "spacer": "spacer",
        "washer": "washer",
        "robotarm": None,  # 複合部品のため複数選択が必要
    }

    return mapping.get(hint_lower)


def infer_parts_from_intent(
    intent: Dict[str, Any],
    vision: Optional[Dict] = None,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    ユーザー意図から複数部品を推定する。

    api_key がある場合はAIを使用し、無い場合はヒューリスティックにフォールバックする。

    引数:
        intent: interact.py からの意図辞書
        vision: 任意のビジョン解析結果
        api_key: AI選択用の OpenAI API キー

    戻り値:
        部品IDとパラメータを含む選択結果のリスト
    """
    if api_key:
        # ライブラリのAI選択を使用
        try:
            from .library import select_parts_for_intent
            result = select_parts_for_intent(intent, api_key, vision)
            return [
                {
                    "part_id": sp.part_id,
                    "parameters": sp.parameters,
                    "confidence": sp.confidence,
                    "reasoning": sp.reasoning,
                    "quantity": sp.quantity
                }
                for sp in result.parts
            ]
        except Exception:
            # AI失敗時はヒューリスティックにフォールバック
            pass

    # 簡易ヒューリスティックにフォールバック
    return _heuristic_part_selection(intent, vision)


def _heuristic_part_selection(
    intent: Dict[str, Any],
    vision: Optional[Dict]
) -> List[Dict[str, Any]]:
    """AIなしのフォールバック用ヒューリスティック部品選択。"""
    parts = []

    # ビジョンのヒントを確認
    if vision:
        vision_result = infer_part_from_vision(vision)
        part_id = _map_hint_to_library_id(vision_result.get("label", ""))
        if part_id:
            parts.append({
                "part_id": part_id,
                "parameters": {},
                "confidence": vision_result["confidence"],
                "reasoning": "Inferred from vision analysis",
                "quantity": 1
            })

    # 機構タイプを確認
    mechanism = (intent.get("mechanism_type") or "").lower()

    if "gear" in mechanism or "歯車" in mechanism:
        gear_params = _infer_gear_params(intent)
        if not any(p["part_id"] == "spur_gear" for p in parts):
            parts.append({
                "part_id": "spur_gear",
                "parameters": gear_params,
                "confidence": 0.6,
                "reasoning": "Mechanism type indicates gear",
                "quantity": 2
            })

    if "shaft" in mechanism or "軸" in mechanism:
        if not any(p["part_id"] == "shaft" for p in parts):
            parts.append({
                "part_id": "shaft",
                "parameters": {},
                "confidence": 0.5,
                "reasoning": "Mechanism requires shaft",
                "quantity": 1
            })

    # 接続方法を確認
    connections = (intent.get("connections") or "").lower()
    if "bolt" in connections or "ボルト" in connections:
        parts.append({
            "part_id": "bolt",
            "parameters": {"size": "M5"},
            "confidence": 0.7,
            "reasoning": "Connection method requires bolts",
            "quantity": 4
        })
        parts.append({
            "part_id": "nut",
            "parameters": {"size": "M5"},
            "confidence": 0.7,
            "reasoning": "Bolts require nuts",
            "quantity": 4
        })

    return parts


def _infer_gear_params(intent: Dict[str, Any]) -> Dict[str, Any]:
    """意図フィールドから歯車パラメータを推定する。"""
    params = {}

    if intent.get("gear_module"):
        try:
            params["module"] = float(intent["gear_module"])
        except (ValueError, TypeError):
            pass

    if intent.get("gear_teeth_count"):
        try:
            params["teeth_count"] = int(intent["gear_teeth_count"])
        except (ValueError, TypeError):
            pass

    if intent.get("gear_width"):
        try:
            params["face_width_mm"] = float(intent["gear_width"])
        except (ValueError, TypeError):
            pass

    return params
