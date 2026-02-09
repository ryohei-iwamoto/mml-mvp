import math

HOLE_CLEARANCE_MM = {
    "M3": 3.4,
    "M4": 4.5,
    "M5": 5.5,
    "M6": 6.6,
    "M8": 9.0,
}

INTENT_FIELDS = [
    ("intent_summary", "この部品で実現したいこと（概要）"),
    ("function_primary", "主機能（何をする部品か）"),
    ("function_secondary", "副機能・付加機能"),
    ("material_intent", "材質の希望（例: 金属/樹脂/ゴムなど）"),
    ("surface_feel", "手触り・表面の質感（例: ざらざら/滑らか/柔らかいなど）"),
    ("surface_finish", "表面仕上げ（例: 研磨/塗装/アルマイト/メッキなど）"),
    ("surface_color", "色や外観の希望"),
    ("texture_pattern", "表面テクスチャやパターンの有無"),
    ("edge_treatment", "エッジ処理（面取り/角R/バリ取りなど）"),
    ("safety_edges", "安全のために丸めたい箇所"),
    ("mechanism_type", "機構の種類（例: 歯車/リンク/カム）"),
    ("motion_type", "運動の種類（回転/直動/揺動など）"),
    ("motion_axis", "運動軸・方向"),
    ("motion_range", "運動範囲（角度/ストロークなど）"),
    ("motion_speed", "目標速度・応答性"),
    ("motion_smoothness", "滑らかさ/バックラッシュ許容"),
    ("motion_control", "制御方式の前提（手動/モータ/スプリング等）"),
    ("force_direction", "力のかかる方向（例: 上下/左右/ねじりなど）"),
    ("force_type", "どんな力がかかるか（例: 回転/衝撃/せん断/引張/圧縮など）"),
    ("force_magnitude", "力の大きさの目安"),
    ("torque_range", "トルクの想定範囲"),
    ("shock_loads", "衝撃・瞬間荷重の有無"),
    ("vibration", "振動の有無と程度"),
    ("fatigue", "繰り返し荷重・疲労の想定"),
    ("moving_parts", "動く部分"),
    ("fixed_parts", "固定される部分"),
    ("interfaces", "取り付け面・接触面・基準面"),
    ("alignment", "芯出し・位置合わせの要件"),
    ("clearances", "クリアランス/干渉回避の要件"),
    ("tolerances", "許容誤差・公差の考え方"),
    ("assembly_method", "組立方法・組立順序の想定"),
    ("disassembly", "分解・メンテナンス性の要件"),
    ("fastening", "締結方法（ボルト/溶接/接着/嵌合など）"),
    ("wiring_routing", "配線・配管の取り回し要件"),
    ("connections", "他部品との接続方法（ボルト、溶接など）"),
    ("load_cases", "力がかかる箇所・方向・大きさの想定"),
    ("torque", "トルク条件"),
    ("speed", "回転/移動速度"),
    ("duty_cycle", "稼働率・使用頻度"),
    ("supports", "支える対象・支点・支持条件"),
    ("power_transmission", "動力伝達の前提（伝達経路/方式）"),
    ("constraints_intent", "成立条件・制約（強度、加工、規格など）"),
    ("safety_factor", "安全率の要求"),
    ("accuracy", "精度要求（位置/角度/速度など）"),
    ("noise", "騒音/振動に関する要求"),
    ("lubrication", "潤滑条件"),
    ("environment", "使用環境（温度/腐食/粉塵/水など）"),
    ("regulations", "規格・法規制"),
    ("analysis_targets", "解析対象（強度/変形/疲労など）"),
    ("verification", "検証方法・試験条件"),
    ("subcomponents", "構成部品（将来的な複数図面化の想定）"),
    ("notes_intent", "その他メモ"),
    ("lifecycle", "製品寿命・耐用年数の想定"),
    ("reliability", "信頼性・故障許容の考え方"),
    ("maintenance", "メンテナンス頻度・保守要件"),
    ("cost_target", "コスト目標・優先順位"),
    ("weight_limit", "重量制約・軽量化の優先度"),
    ("size_limit", "サイズ制約・外形制限"),
    ("noise_limit", "騒音レベルの制約"),
    ("heat_generation", "発熱や熱対策の要件"),
    ("cooling", "冷却方法の前提"),
    ("chemical_resistance", "薬品/油/水への耐性"),
    ("electric_isolation", "電気絶縁の要件"),
    ("grounding", "接地・静電気対策"),
    ("ergonomics", "人の操作性/触感の要件"),
    ("aesthetics", "見た目の優先度"),
    ("modularity", "モジュール性・交換性の要件"),
    ("compatibility", "既存部品との互換性"),
    ("standards_fit", "規格品の流用有無"),
    ("transportation", "搬送・運搬時の条件"),
    ("storage", "保管条件"),
    ("gear_module", "歯車: モジュール"),
    ("gear_teeth_count", "歯車: 歯数"),
    ("gear_pressure_angle", "歯車: 圧力角"),
    ("gear_width", "歯車: 歯幅"),
    ("gear_backlash", "歯車: バックラッシュ"),
    ("gear_material", "歯車: 材質の希望"),
    ("gear_noise", "歯車: 騒音/振動の許容"),
    ("gear_lubrication", "歯車: 潤滑の要否"),
]


def _hole_radii_px(vision):
    return [h.get("radius_px") for h in vision.get("holes", []) if h.get("radius_px") is not None]


def _holes_vary(radii):
    if len(radii) < 2:
        return False
    mean = sum(radii) / len(radii)
    if mean == 0:
        return False
    variance = sum((r - mean) ** 2 for r in radii) / len(radii)
    return math.sqrt(variance) / mean > 0.08


def _parse_hole_standard(value):
    if not value:
        return None
    key = value.strip().upper()
    if key in HOLE_CLEARANCE_MM:
        return key
    return None


def _parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"y", "yes", "true", "1"}:
        return True
    if text in {"n", "no", "false", "0"}:
        return False
    return None


def _has_value(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def build_questions(vision, params):
    questions = []

    if vision.get("holes") and not params.get("hole_standard") and params.get("hole_diameter_mm") is None:
        questions.append(
            {
                "id": "hole_standard",
                "text": "穴の規格は？ (M3/M4/M5/M6/M8) もしくは空欄",
                "type": "str",
            }
        )
        questions.append({"id": "hole_diameter_mm", "text": "穴の直径(mm) ※規格が不明な場合", "type": "float"})

    radii = _hole_radii_px(vision)
    if _holes_vary(radii) and params.get("unify_holes") is None:
        questions.append({"id": "unify_holes", "text": "穴サイズを統一しますか？ (y/n)", "type": "bool"})

    if params.get("thickness_mm") is None:
        questions.append({"id": "thickness_mm", "text": "板厚は何mmですか？", "type": "float"})

    if vision.get("bend_lines"):
        if params.get("bend_angle_deg") is None:
            questions.append({"id": "bend_angle_deg", "text": "曲げ角度は何度ですか？", "type": "float"})
        if params.get("bend_radius_mm") is None:
            questions.append({"id": "bend_radius_mm", "text": "曲げ内Rは何mmですか？", "type": "float"})

    return questions


def build_model_questions(vision, params, inferred_part=None):
    questions = []
    if inferred_part:
        questions.append(
            {
                "id": "part_type_confirm",
                "text": f"図形から推定した部品種別は「{inferred_part}」です。合っていますか？(yes/no/別名)",
                "type": "text",
            }
        )

    has_holes = bool(vision.get("holes"))
    has_bend = bool(vision.get("bend_lines"))

    if inferred_part and str(inferred_part).lower() == "robotarm":
        if not _has_value(params.get("arm_config")):
            questions.append(
                {
                    "id": "arm_config",
                    "text": "ロボットアームの自由度(関節数)、駆動方式、到達距離、想定荷重を教えてください。",
                    "type": "text",
                }
            )

    for qid, text in INTENT_FIELDS:
        if not _has_value(params.get(qid)):
            questions.append({"id": qid, "text": text, "type": "text"})

    if has_holes and not _has_value(params.get("connections")):
        questions.append({"id": "connections", "text": "他部品との接続方法（ボルト、溶接など）", "type": "text"})
    if has_bend and not _has_value(params.get("process_intent_detail")):
        questions.append({"id": "process_intent_detail", "text": "曲げ加工に関する意図・制約", "type": "text"})

    return questions


def resolve_params(vision, params, prompt_fn=None, include_intent=False, inferred_part=None):
    if include_intent:
        questions = build_model_questions(vision, params, inferred_part=inferred_part)
    else:
        questions = build_questions(vision, params)
    answers = []
    answer_map = {}

    for q in questions:
        value = None
        if q["id"] in params and params.get(q["id"]) is not None:
            value = params.get(q["id"])
        elif prompt_fn:
            value = prompt_fn(q)
        answers.append({"id": q["id"], "value": value})
        answer_map[q["id"]] = value

    px_to_mm = params.get("px_to_mm")
    plate_width_mm = params.get("plate_width_mm")

    outline = vision.get("outline", {}).get("points_px", [])
    if px_to_mm is None and plate_width_mm and outline:
        xs = [p[0] for p in outline]
        width_px = max(xs) - min(xs)
        if width_px > 0:
            px_to_mm = float(plate_width_mm) / float(width_px)

    if px_to_mm is None:
        px_to_mm = 1.0

    hole_standard = params.get("hole_standard")
    if hole_standard is None:
        hole_standard = _parse_hole_standard(answer_map.get("hole_standard"))
    hole_diameter_mm = params.get("hole_diameter_mm")
    if hole_diameter_mm is None:
        hole_diameter_mm = answer_map.get("hole_diameter_mm")
    if hole_standard:
        hole_diameter_mm = HOLE_CLEARANCE_MM.get(hole_standard)
    elif hole_diameter_mm is not None:
        hole_diameter_mm = float(hole_diameter_mm)
        hole_standard = "custom"

    unify_holes = params.get("unify_holes")
    if unify_holes is None:
        unify_holes = _parse_bool(answer_map.get("unify_holes"))

    thickness_mm = params.get("thickness_mm")
    if thickness_mm is None:
        thickness_mm = answer_map.get("thickness_mm")

    bend_angle_deg = params.get("bend_angle_deg")
    if bend_angle_deg is None:
        bend_angle_deg = answer_map.get("bend_angle_deg")

    bend_radius_mm = params.get("bend_radius_mm")
    if bend_radius_mm is None:
        bend_radius_mm = answer_map.get("bend_radius_mm")

    hole_diameters_mm = None
    if vision.get("holes"):
        if hole_diameter_mm is not None and (unify_holes is None or unify_holes is True):
            hole_diameters_mm = [float(hole_diameter_mm) for _ in vision.get("holes")]
        else:
            hole_diameters_mm = []
            for h in vision.get("holes", []):
                hole_diameters_mm.append(float(h.get("radius_px", 0)) * 2 * float(px_to_mm))

    intent = {
        "summary": answer_map.get("intent_summary") or params.get("intent_summary"),
        "inferred_part": params.get("inferred_part"),
        "part_type_confirm": answer_map.get("part_type_confirm") or params.get("part_type_confirm"),
    }

    for qid, _ in INTENT_FIELDS:
        if qid in {"intent_summary"}:
            continue
        intent[qid] = answer_map.get(qid) or params.get(qid)
    intent["process_detail"] = answer_map.get("process_intent_detail") or params.get("process_intent_detail")

    resolved = {
        "px_to_mm": float(px_to_mm),
        "hole_standard": hole_standard,
        "hole_diameter_mm": hole_diameter_mm,
        "hole_diameters_mm": hole_diameters_mm,
        "thickness_mm": float(thickness_mm) if thickness_mm is not None else None,
        "bend_angle_deg": float(bend_angle_deg) if bend_angle_deg is not None else None,
        "bend_radius_mm": float(bend_radius_mm) if bend_radius_mm is not None else None,
        "unify_holes": unify_holes,
        "intent": intent,
    }

    chat = {"mode": "rule", "questions": questions, "answers": answers}
    return resolved, chat
