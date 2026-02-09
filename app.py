import json
import os
import math
import copy
import re
import ast

from dotenv import load_dotenv
from flask import Flask, render_template, request, send_from_directory, redirect, url_for
from openai import OpenAI
import trimesh
from werkzeug.utils import secure_filename

from mml.ai_vision import run_ai_vision
from mml.draw import draw_dxf, draw_png
from mml.emit import emit_mml
from mml.intent import infer_part_from_vision
from mml.interact import HOLE_CLEARANCE_MM, build_model_questions
from mml.pipeline import run_pipeline
from mml.stl import write_stl
from mml.utils import ensure_dir, new_run_id, read_json, write_json
from mml.vision import normalize_vision, run_vision

load_dotenv()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(APP_ROOT, "outputs")
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

app = Flask(__name__)

MANDATORY_ABSTRACT_IDS = [
    "intent_summary",
    "function_primary",
    "mechanism_type",
    "motion_type",
    "connections",
    "load_cases",
    "constraints_intent",
]


def _get_form_float(name):
    value = request.form.get(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _get_form_int(name):
    value = request.form.get(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _get_form_str(name):
    value = request.form.get(name, "").strip()
    return value or None


def _parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"y", "yes", "true", "1", "on"}:
        return True
    if text in {"n", "no", "false", "0"}:
        return False
    return None


@app.route("/")
def index():
    return render_template("index.html")


def _vision_from_upload(upload, use_ai):
    if upload is None or upload.filename == "":
        return None, None
    filename = secure_filename(upload.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTS:
        raise ValueError("未対応のファイル形式です。")

    run_id = new_run_id()
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    ensure_dir(run_dir)

    input_path = os.path.join(run_dir, f"input{ext.lower()}")
    upload.save(input_path)

    if use_ai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY が未設定です。.env を作成してください。")
        model = os.getenv("OPENAI_MODEL")
        vision = run_ai_vision(input_path, api_key=api_key, model=model)
    else:
        vision = run_vision(input_path)

    return run_dir, {"input_path": input_path, "vision": normalize_vision(vision)}


def _empty_vision():
    return {"outline": {"type": "polygon", "points_px": []}, "holes": [], "bend_lines": [], "notes_regions": []}


def _prompt_from_form(form):
    def _prompt(q):
        raw = form.get(q["id"], "")
        if raw == "":
            return None
        if q.get("type") == "float":
            try:
                return float(raw)
            except ValueError:
                return None
        if q.get("type") == "bool":
            return _parse_bool(raw)
        return raw

    return _prompt


def _has_answer(answers_state, key):
    value = answers_state.get(key)
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True


def _normalize_text(text):
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def _is_similar_text(a, b):
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    # 言い換えを拾うための簡易類似度ヒューリスティック。
    ratio = 0.0
    try:
        import difflib

        ratio = difflib.SequenceMatcher(a=a_norm, b=b_norm).ratio()
    except Exception:
        ratio = 0.0
    return ratio >= 0.82


def _choose_next_question(questions, answers_state, inferred_part, mandatory_ids=None):
    remaining = [q for q in questions if not _has_answer(answers_state, q["id"])]
    if not remaining:
        return None
    if mandatory_ids:
        mandatory_remaining = [q for q in remaining if q["id"] in mandatory_ids]
        if mandatory_remaining:
            return mandatory_remaining[0]

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return remaining[0]

    client = OpenAI(api_key=api_key)
    payload = {
        "inferred_part": inferred_part,
        "answers": answers_state,
        "remaining": [{"id": q["id"], "text": q["text"]} for q in remaining],
        "instruction": "次に聞くべき質問IDを1つ選び、JSONで返してください: {\"next_id\": \"...\"}",
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "抽象設計の次の質問を1つ選びます。日本語で判断してください。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        next_id = data.get("next_id")
        for q in remaining:
            if q["id"] == next_id:
                return q
    except Exception:
        pass
    return remaining[0]


def _ai_generate_next_abstract_question(answers_state, inferred_part, asked_ids, asked_texts):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    missing_mandatory = [mid for mid in MANDATORY_ABSTRACT_IDS if not _has_answer(answers_state, mid)]
    payload = {
        "inferred_part": inferred_part,
        "asked": asked_ids,
        "asked_texts": asked_texts,
        "answers": answers_state,
        "mandatory_ids": MANDATORY_ABSTRACT_IDS,
        "missing_mandatory": missing_mandatory,
        "instruction": (
            "抽象設計に必要な質問を動的に判断して、次の質問を1つだけ生成してください。"
            "材料・加工方法・正確な寸法は聞かないでください。"
            "missing_mandatory がある場合は必ずそのいずれかを優先してください。"
            "asked_textsに意味的に近い質問は出さないでください。"
            "質問が十分だと判断した場合は {\"done\": true} を返してください。"
            "返答はJSONのみ: {\"id\": \"...\", \"text\": \"...\", \"type\": \"text\"} または {\"done\": true}."
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "次の抽象設計の質問を日本語で1つ選びます。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return None
        data = json.loads(text)
        if data.get("done") is True:
            return {"done": True}
        if data.get("id") and data.get("text"):
            return {"id": data["id"], "text": data["text"], "type": data.get("type", "text")}
    except Exception:
        return None
    return None


def _ai_classify_message(text, inferred_part):
    if not text:
        return "answer"
    if text.strip().endswith(("?", "？")):
        return "question"
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "answer"
    client = OpenAI(api_key=api_key)
    payload = {
        "inferred_part": inferred_part,
        "message": text,
        "instruction": "このメッセージが質問なら question、回答なら answer をJSONで返す。",
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "質問か回答かを判定します。日本語で判断してください。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content.strip())
        if data.get("label") in {"question", "answer"}:
            return data["label"]
    except Exception:
        return "answer"
    return "answer"


def _ai_answer_user_question(message, inferred_part, answers_state):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "現状の情報では判断が難しいため、目的や制約を教えてください。"
    client = OpenAI(api_key=api_key)
    payload = {
        "inferred_part": inferred_part,
        "answers": answers_state,
        "message": message,
        "instruction": (
            "抽象設計の相談に短く答えてください。"
            "材料・加工方法・正確な寸法は避け、目的や機能に沿った助言を返してください。"
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "抽象設計の相談に答えるアシスタントです。"},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return response.choices[0].message.content.strip() or "もう少し具体的に教えてください。"
    except Exception:
        return "もう少し具体的に教えてください。"


def _ai_suggest_subcomponents(inferred_part, answers_state, supplemental_text=None):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or not inferred_part:
        return []
    client = OpenAI(api_key=api_key)
    payload = {
        "part": inferred_part,
        "answers": answers_state,
        "supplemental": supplemental_text or "",
        "instruction": (
            "List the mechanical subcomponents required to build the part. "
            "Exclude control/electrical/software items. "
            "Use short names like Base, Joint, Link, EndEffector, Actuator, MotorMount, Gear, Shaft, Bearing. "
            "Return JSON: {\"subcomponents\": [..]}"
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You list required mechanical components."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text) if text else {}
        items = data.get("subcomponents", [])
        return _normalize_subcomponents(items)
    except Exception:
        return []


def _default_robotarm_config():
    return {
        "joint_count": 2,
        "drive_type": "gear",
        "reach_mm": 300,
        "payload_kg": 0.5,
        "notes": "auto",
    }


def _robotarm_config_summary(cfg):
    if not isinstance(cfg, dict):
        return ""
    return f"J{cfg.get('joint_count')} / {cfg.get('drive_type')} / reach {cfg.get('reach_mm')}mm / payload {cfg.get('payload_kg')}kg"


def _ai_suggest_robotarm_config(inferred_part, answers_state, supplemental_text=None):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or not inferred_part:
        return _default_robotarm_config()
    if str(inferred_part).strip().lower() != "robotarm":
        return _default_robotarm_config()
    client = OpenAI(api_key=api_key)
    payload = {
        "part": inferred_part,
        "answers": answers_state,
        "supplemental": supplemental_text or "",
        "instruction": (
            "Propose a minimal robot arm configuration. "
            "Return JSON with joint_count (int), drive_type (gear/belt/direct), reach_mm (number), payload_kg (number)."
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You propose robot arm configuration."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text) if text else {}
        return {
            "joint_count": int(data.get("joint_count", 2)),
            "drive_type": str(data.get("drive_type", "gear")),
            "reach_mm": float(data.get("reach_mm", 300)),
            "payload_kg": float(data.get("payload_kg", 0.5)),
            "notes": "ai",
        }
    except Exception:
        return _default_robotarm_config()


def _ai_parse_robotarm_config(text_value, inferred_part=None):
    if not text_value:
        return _default_robotarm_config()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return _default_robotarm_config()
    client = OpenAI(api_key=api_key)
    payload = {
        "text": text_value,
        "part": inferred_part or "RobotArm",
        "instruction": (
            "Extract robot arm config. Return JSON with joint_count (int), drive_type (gear/belt/direct), "
            "reach_mm (number), payload_kg (number)."
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract robot arm configuration."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text) if text else {}
        return {
            "joint_count": int(data.get("joint_count", 2)),
            "drive_type": str(data.get("drive_type", "gear")),
            "reach_mm": float(data.get("reach_mm", 300)),
            "payload_kg": float(data.get("payload_kg", 0.5)),
            "notes": "parsed",
        }
    except Exception:
        return _default_robotarm_config()


def _default_robotarm_dims():
    return {
        "link_length_mm": 140.0,
        "link_width_mm": 26.0,
        "link_hole_offset_mm": 16.0,
        "joint_outer_diameter_mm": 36.0,
        "joint_hole_diameter_mm": 8.0,
        "gear_outer_diameter_mm": 36.0,
        "gear_bore_diameter_mm": 8.0,
        "shaft_diameter_mm": 8.0,
        "bearing_outer_diameter_mm": 30.0,
        "bearing_inner_diameter_mm": 12.0,
        "motor_outer_diameter_mm": 40.0,
        "motor_mount_width_mm": 70.0,
        "motor_mount_height_mm": 50.0,
        "base_width_mm": 100.0,
        "base_height_mm": 70.0,
    }


def _ai_refine_robotarm_dims(inferred_part, intent, max_iters=6):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    dims = _default_robotarm_dims()
    log = []
    if not api_key or str(inferred_part).strip().lower() != "robotarm":
        return dims, log

    client = OpenAI(api_key=api_key)
    for _ in range(max_iters):
        payload = {
            "intent": intent,
            "current_dims": dims,
            "instruction": (
                "You are co-designing a robot arm. "
                "Adjust dimensions so parts can be assembled: joint holes match shafts, "
                "gear bore matches shaft, link holes match joints, and sizes are plausible. "
                "Return JSON with: {\"status\": \"ok\"|\"adjust\", \"dims\": {...}, \"notes\": \"...\"}."
            ),
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You refine mechanical dimensions."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            data = json.loads(text) if text else {}
            status = str(data.get("status", "adjust")).lower()
            new_dims = data.get("dims") or {}
            if isinstance(new_dims, dict):
                dims.update(new_dims)
            log.append({"status": status, "notes": data.get("notes")})
            if status == "ok":
                break
        except Exception as exc:
            log.append({"status": "error", "notes": str(exc)})
            break
    return dims, log





def _canonical_subcomponent_name(text):
    if not text:
        return None
    name = str(text).strip().lower()
    if not name:
        return None
    if "base" in name:
        return "base"
    if "joint" in name:
        return "joint"
    if "link" in name or "arm" in name:
        return "link"
    if "end effector" in name or "end_effector" in name or "gripper" in name:
        return "end_effector"
    if "actuator" in name or "motor" in name or "servo" in name:
        return "actuator"
    if "motor_mount" in name or "motor mount" in name or "mount" in name:
        return "motor_mount"
    if "shaft" in name:
        return "shaft"
    if "gear" in name:
        return "gear"
    if "bearing" in name:
        return "bearing"
    if "spacer" in name:
        return "spacer"
    if "bracket" in name:
        return "bracket"
    if "housing" in name or "case" in name:
        return "housing"
    return None


def _normalize_subcomponents(items):
    if not isinstance(items, list):
        return []
    normalized = []
    separators = [",", "\n", "\t", "/", "、", "・", ";", "|"]
    for item in items:
        if isinstance(item, dict):
            candidate = item.get("name") or item.get("type") or item.get("part")
            canon = _canonical_subcomponent_name(candidate)
            if canon:
                normalized.append(canon)
            continue
        text_item = str(item).strip()
        if not text_item:
            continue
        if text_item.startswith("{") and text_item.endswith("}"):
            try:
                data = ast.literal_eval(text_item)
            except Exception:
                data = None
            if isinstance(data, dict):
                candidate = data.get("name") or data.get("type") or data.get("part")
                canon = _canonical_subcomponent_name(candidate)
                if canon:
                    normalized.append(canon)
                continue
        parts = [text_item]
        for sep in separators:
            split_parts = []
            for p in parts:
                split_parts.extend([s.strip() for s in p.split(sep) if s.strip()])
            parts = split_parts
        for part in parts:
            canon = _canonical_subcomponent_name(part)
            if canon:
                normalized.append(canon)
    return normalized

def _ensure_robotarm_components(items, inferred_part):
    if not items or not inferred_part:
        return items
    if str(inferred_part).strip().lower() != "robotarm":
        return items
    counts = {}
    for name in items:
        counts[name] = counts.get(name, 0) + 1
    if counts.get("link", 0) >= 2 and counts.get("joint", 0) >= 2 and "base" in counts and "end_effector" in counts:
        return items
    return [
        "base",
        "joint",
        "link",
        "joint",
        "link",
        "end_effector",
        "actuator",
        "motor_mount",
        "gear",
        "gear",
        "shaft",
        "bearing",
    ]


def _assemble_stl(run_dir, outputs_multi):
    meshes = []
    cursor_x = 0.0
    spacing = 20.0
    for item in outputs_multi:
        files = item.get("files") or {}
        stl_name = files.get("stl")
        if not stl_name:
            continue
        stl_path = os.path.join(run_dir, stl_name)
        if not os.path.exists(stl_path):
            continue
        try:
            mesh = trimesh.load(stl_path, force="mesh")
        except Exception:
            continue
        if mesh.is_empty:
            continue
        bounds = mesh.bounds
        size_x = bounds[1][0] - bounds[0][0]
        mesh.apply_translation((-bounds[0][0] + cursor_x, -bounds[0][1], -bounds[0][2]))
        meshes.append(mesh)
        cursor_x += size_x + spacing
    if not meshes:
        return None
    assembly = trimesh.util.concatenate(meshes)
    out_name = "assembly.stl"
    out_path = os.path.join(run_dir, out_name)
    assembly.export(out_path)
    return out_name




@app.route("/run", methods=["POST"])
def run():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return render_template("index.html", error="画像ファイルを選択してください。")

    filename = secure_filename(upload.filename)
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTS:
        return render_template("index.html", error="未対応のファイル形式です。")

    run_id = new_run_id()
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    ensure_dir(run_dir)

    input_path = os.path.join(run_dir, f"input{ext.lower()}")
    upload.save(input_path)

    part_name = _get_form_str("part_name")
    inferred_part = _get_form_str("inferred_part")
    part_type_confirm = _get_form_str("part_type_confirm")
    if not part_name and inferred_part and (part_type_confirm or "").strip().lower() in {"yes", "y"}:
        part_name = inferred_part
    if not part_name and inferred_part:
        part_name = inferred_part

    params = {
        "part_name": part_name,
        "material": _get_form_str("material"),
        "process": _get_form_str("process"),
        "plate_width_mm": _get_form_float("plate_width_mm"),
        "thickness_mm": _get_form_float("thickness_mm"),
        "hole_standard": _get_form_str("hole_standard"),
        "hole_diameter_mm": _get_form_float("hole_diameter_mm"),
        "bend_angle_deg": _get_form_float("bend_angle_deg"),
        "bend_radius_mm": _get_form_float("bend_radius_mm"),
        "use_ai": request.form.get("use_ai") == "on",
    }

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    try:
        outputs = run_pipeline(input_path, run_dir, params=params, api_key=api_key, model=model)
    except ValueError as exc:
        return render_template("index.html", error=str(exc))

    return render_template(
        "result.html",
        run_id=run_id,
        outputs={
            "vision": os.path.basename(outputs["vision"]),
            "mml": os.path.basename(outputs["mml"]),
            "report": os.path.basename(outputs["report"]),
            "dxf": os.path.basename(outputs["dxf"]),
            "png": os.path.basename(outputs["png"]),
            "stl": os.path.basename(outputs["stl"]),
            "input": os.path.basename(input_path),
        },
    )


@app.route("/outputs/<run_id>/<path:filename>")
def download(run_id, filename):
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    return send_from_directory(run_dir, filename, as_attachment=True)


@app.route("/model", methods=["GET"])
def model_start():
    return render_template("model_start.html")


@app.route("/model/vision", methods=["POST"])
def model_vision():
    use_ai = request.form.get("use_ai") == "on"
    upload = request.files.get("image")
    try:
        run_dir, payload = _vision_from_upload(upload, use_ai)
    except ValueError as exc:
        return render_template("model_start.html", error=str(exc))

    if payload is None:
        run_id = new_run_id()
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        ensure_dir(run_dir)
        payload = {"input_path": None, "vision": _empty_vision()}
    else:
        run_id = os.path.basename(run_dir)

    write_json(os.path.join(run_dir, "vision.json"), payload["vision"])
    inferred = infer_part_from_vision(payload["vision"])
    questions = build_model_questions(payload["vision"], params={}, inferred_part=inferred["label"])
    answers_state = {}
    asked_ids = []
    asked_texts = []
    chat_log = []
    supplemental_text = _get_form_str("supplemental_text")
    if supplemental_text:
        answers_state["notes_intent"] = supplemental_text
    suggested_subs = _ai_suggest_subcomponents(inferred["label"], answers_state, supplemental_text)
    if suggested_subs:
        for q in questions:
            if q.get("id") == "subcomponents":
                q["text"] = f"{q['text']} (Suggested: {', '.join(suggested_subs)} / edit OK)"
                break
    suggested_subcomponents_json = json.dumps(suggested_subs, ensure_ascii=False)
    suggested_arm_config = _ai_suggest_robotarm_config(inferred["label"], answers_state, supplemental_text)
    suggested_arm_config_json = json.dumps(suggested_arm_config, ensure_ascii=False)
    if suggested_arm_config:
        for q in questions:
            if q.get("id") == "arm_config":
                q["text"] = f"{q['text']} (Suggested: {_robotarm_config_summary(suggested_arm_config)})"
                break
    first_question = None
    ai_question = _ai_generate_next_abstract_question(answers_state, inferred["label"], asked_ids, asked_texts)
    if isinstance(ai_question, dict) and ai_question.get("done"):
        if any(not _has_answer(answers_state, mid) for mid in MANDATORY_ABSTRACT_IDS):
            ai_question = None
            first_question = _choose_next_question(questions, answers_state, inferred["label"], MANDATORY_ABSTRACT_IDS)
        else:
            first_question = None
    elif isinstance(ai_question, dict) and ai_question.get("id"):
        first_question = ai_question
    elif questions:
        for q in questions:
            if q["id"] == "intent_summary":
                first_question = q
                break
        if first_question is None:
            first_question = questions[0]
    if first_question and first_question.get("id"):
        asked_ids.append(first_question["id"])
    if first_question and first_question.get("text"):
        asked_texts.append(first_question["text"])
    api_error = not os.getenv("OPENAI_API_KEY")
    return render_template(
        "model_chat.html",
        run_id=run_id,
        questions=questions,
        current_question=first_question,
        answers_json=json.dumps(answers_state, ensure_ascii=False),
        questions_json=json.dumps(questions, ensure_ascii=False),
        asked_json=json.dumps(asked_ids, ensure_ascii=False),
        asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
        chat_json=json.dumps(chat_log, ensure_ascii=False),
        chat_log=chat_log,
        inferred_part=inferred["label"],
        inferred_confidence=inferred["confidence"],
        part_name=_get_form_str("part_name"),
        suggested_subcomponents_json=suggested_subcomponents_json,
        suggested_arm_config_json=suggested_arm_config_json,
        api_error=api_error,
    )


@app.route("/model/vision", methods=["GET"])
def model_vision_get():
    return redirect(url_for("model_start"))


@app.route("/model/emit", methods=["POST"])
def model_emit():
    run_id = request.form.get("run_id")
    if not run_id:
        return render_template("model_start.html", error="run id がありません。")
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    vision_path = os.path.join(run_dir, "vision.json")
    if not os.path.exists(vision_path):
        return render_template("model_start.html", error="vision.json が見つかりません。")
    vision = normalize_vision(read_json(vision_path))
    write_json(vision_path, vision)

    part_name = _get_form_str("part_name")
    inferred_part = _get_form_str("inferred_part")
    part_type_confirm = _get_form_str("part_type_confirm")
    if not part_name and inferred_part and (part_type_confirm or "").strip().lower() in {"yes", "y"}:
        part_name = inferred_part

    questions_json = request.form.get("questions_json", "[]")
    answers_json = request.form.get("answers_json", "{}")
    asked_json = request.form.get("asked_json", "[]")
    asked_texts_json = request.form.get("asked_texts_json", "[]")
    chat_json = request.form.get("chat_json", "[]")
    suggested_subcomponents_json = request.form.get("suggested_subcomponents_json")
    suggested_arm_config_json = request.form.get("suggested_arm_config_json")
    try:
        questions = json.loads(questions_json)
    except json.JSONDecodeError:
        questions = []
    try:
        answers_state = json.loads(answers_json)
    except json.JSONDecodeError:
        answers_state = {}
    try:
        asked_ids = json.loads(asked_json)
    except json.JSONDecodeError:
        asked_ids = []
    try:
        asked_texts = json.loads(asked_texts_json)
    except json.JSONDecodeError:
        asked_texts = []
    try:
        chat_log = json.loads(chat_json)
    except json.JSONDecodeError:
        chat_log = []

    suggested_subs = []
    if suggested_subcomponents_json:
        try:
            suggested_subs = json.loads(suggested_subcomponents_json)
        except json.JSONDecodeError:
            suggested_subs = []
    suggested_arm_config = {}
    if suggested_arm_config_json:
        try:
            suggested_arm_config = json.loads(suggested_arm_config_json)
        except json.JSONDecodeError:
            suggested_arm_config = {}

    current_id = request.form.get("current_id")
    current_value = request.form.get("current_value")
    chat_action = request.form.get("chat_action", "answer")
    if chat_action == "auto":
        chat_action = _ai_classify_message(current_value or "", inferred_part)

    if chat_action == "question":
        if current_value:
            current_text = request.form.get("current_text")
            if current_text:
                chat_log.append({"role": "assistant", "text": current_text})
            chat_log.append({"role": "user", "text": current_value})
            reply = _ai_answer_user_question(current_value, inferred_part, answers_state)
            chat_log.append({"role": "assistant", "text": reply})
        current_text = request.form.get("current_text")
        current_type = request.form.get("current_type")
        current_question = None
        if current_id and current_text:
            current_question = {"id": current_id, "text": current_text, "type": current_type or "text"}
        return render_template(
            "model_chat.html",
            run_id=run_id,
            questions=questions,
            current_question=current_question,
            answers_json=json.dumps(answers_state, ensure_ascii=False),
            questions_json=json.dumps(questions, ensure_ascii=False),
            asked_json=json.dumps(asked_ids, ensure_ascii=False),
            asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
            chat_json=json.dumps(chat_log, ensure_ascii=False),
            chat_log=chat_log,
            inferred_part=inferred_part,
            inferred_confidence=request.form.get("inferred_confidence") or "",
            part_name=part_name,
            suggested_subcomponents_json=suggested_subcomponents_json,
            suggested_arm_config_json=suggested_arm_config_json,
        )

    if current_id:
        current_text = request.form.get("current_text")
        if current_text:
            chat_log.append({"role": "assistant", "text": current_text})
            if current_text not in asked_texts:
                asked_texts.append(current_text)
        if current_value:
            chat_log.append({"role": "user", "text": current_value})
        answers_state[current_id] = current_value
        if current_id not in asked_ids:
            asked_ids.append(current_id)
    if request.form.get("skip_all") != "1":
        next_q = _ai_generate_next_abstract_question(answers_state, inferred_part, asked_ids, asked_texts)
        if not next_q:
            next_q = _choose_next_question(questions, answers_state, inferred_part, MANDATORY_ABSTRACT_IDS)
        if isinstance(next_q, dict) and next_q.get("done"):
            next_q = None
        if next_q and next_q.get("text"):
            is_dup = any(_is_similar_text(next_q["text"], t) for t in asked_texts)
            if is_dup:
                next_q = _choose_next_question(questions, answers_state, inferred_part, MANDATORY_ABSTRACT_IDS)
                if next_q and next_q.get("text"):
                    is_dup = any(_is_similar_text(next_q["text"], t) for t in asked_texts)
                    if is_dup:
                        next_q = None
        if next_q:
            return render_template(
                "model_chat.html",
                run_id=run_id,
                questions=questions,
                current_question=next_q,
                answers_json=json.dumps(answers_state, ensure_ascii=False),
                questions_json=json.dumps(questions, ensure_ascii=False),
                asked_json=json.dumps(asked_ids, ensure_ascii=False),
                asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
                chat_json=json.dumps(chat_log, ensure_ascii=False),
                chat_log=chat_log,
                inferred_part=inferred_part,
                inferred_confidence=request.form.get("inferred_confidence") or "",
                part_name=part_name,
                suggested_subcomponents_json=suggested_subcomponents_json,
                suggested_arm_config_json=suggested_arm_config_json,
            )

    params = {"part_name": part_name, "inferred_part": inferred_part, "part_type_confirm": part_type_confirm}
    if (not answers_state.get("subcomponents")) and suggested_subs:
        answers_state["subcomponents"] = suggested_subs
    if (not answers_state.get("arm_config")) and suggested_arm_config:
        answers_state["arm_config"] = suggested_arm_config
    params.update(answers_state)
    mml, report = emit_mml(

        vision,
        params,
        os.path.basename(vision_path),
        prompt_fn=None,
        include_intent=True,
        inferred_part=inferred_part,
    )
    if mml.get("intent") is not None and not mml["intent"].get("subcomponents"):
        inferred = inferred_part or mml["intent"].get("inferred_part") or ""
        subs = _infer_subcomponents(mml["intent"], inferred)
        if subs:
            mml["intent"]["subcomponents"] = subs
        elif inferred.lower() == "robotarm":
            mml["intent"]["subcomponents"] = ["base", "joint", "link", "joint", "link", "end_effector", "actuator"]
    if mml.get("intent") is not None:
        if not mml["intent"].get("arm_config"):
            arm_text = answers_state.get("arm_config")
            if isinstance(arm_text, dict):
                mml["intent"]["arm_config"] = arm_text
            else:
                mml["intent"]["arm_config"] = _ai_parse_robotarm_config(arm_text, inferred_part)
        if str((mml["intent"].get("inferred_part") or inferred_part or "")).lower() == "robotarm":
            dims, dim_log = _ai_refine_robotarm_dims(inferred_part, mml["intent"])
            mml["intent"]["arm_dims"] = dims
            mml["intent"]["arm_dims_log"] = dim_log
        normalized = _normalize_subcomponents(mml["intent"].get("subcomponents"))
        if normalized:
            normalized = _ensure_robotarm_components(
                normalized, inferred_part or mml["intent"].get("inferred_part")
            )
            mml["intent"]["subcomponents"] = normalized
        if (mml.get("part") in {None, "", "Unknown"}) and mml["intent"].get("inferred_part"):
            mml["part"] = mml["intent"]["inferred_part"]
    if "material" in mml:
        del mml["material"]
    if "process" in mml:
        del mml["process"]
    mml_path = os.path.join(run_dir, "mml.json")
    report_path = os.path.join(run_dir, "report.json")
    write_json(mml_path, mml)
    write_json(report_path, report)

    return render_template(
        "model_result.html",
        run_id=run_id,
        outputs={
            "vision": "vision.json",
            "mml": "mml.json",
            "report": "report.json",
        },
    )


@app.route("/model/emit", methods=["GET"])
def model_emit_get():
    return redirect(url_for("model_start"))


def _is_gear(mml):
    part = (mml.get("part") or "").lower()
    intent = mml.get("intent", {}) or {}
    inferred = (intent.get("inferred_part") or "").lower()
    confirmed = (intent.get("part_type_confirm") or "").lower()
    return "gear" in part or inferred == "gear" or confirmed == "gear"


def _is_template_part(mml):
    part = (mml.get("part") or "").lower()
    tokens = [
        "base",
        "joint",
        "link",
        "end_effector",
        "gripper",
        "actuator",
        "motor",
        "servo",
        "shaft",
        "housing",
        "motor_mount",
        "mount",
        "gear",
        "bearing",
        "spacer",
        "bracket",
    ]
    return any(token in part for token in tokens)


def _outline_bounds(points):
    if not points:
        return None
    xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _estimate_outline_size(mml):
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", []) or []
    bounds = _outline_bounds(outline)
    if not bounds:
        return None, None
    min_x, min_y, max_x, max_y = bounds
    return round(max_x - min_x, 3), round(max_y - min_y, 3)


def _estimate_outer_diameter(mml):
    w, h = _estimate_outline_size(mml)
    if w is None or h is None:
        return None
    return round(max(w, h), 3)


def _estimate_bore_diameter(mml):
    holes = mml.get("geometry", {}).get("holes", []) or []
    if not holes:
        return None
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", []) or []
    bounds = _outline_bounds(outline)
    if not bounds:
        return None
    min_x, min_y, max_x, max_y = bounds
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    best = None
    for h in holes:
        center = h.get("center_mm")
        dia = h.get("diameter_mm")
        if not center or dia is None:
            continue
        dist = abs(center[0] - cx) + abs(center[1] - cy)
        if best is None or dist < best[0]:
            best = (dist, dia)
    if best:
        return round(best[1], 3)
    return None


def _draw_questions(mml):
    questions = []

    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    if _is_gear(mml):
        outer_est = _estimate_outer_diameter(mml)
        bore_est = _estimate_bore_diameter(mml)
        questions.append(
            {"id": "outer_diameter_mm", "text": "外径(mm)", "type": "float", "default": outer_est}
        )
        questions.append(
            {"id": "bore_diameter_mm", "text": "内径(mm)", "type": "float", "default": bore_est}
        )
        questions.append({"id": "teeth_count", "text": "歯数", "type": "int"})
    else:
        if not _is_template_part(mml):
            width_est, height_est = _estimate_outline_size(mml)
            questions.append(
                {"id": "outline_width_mm", "text": "外形の幅(mm)", "type": "float", "default": width_est}
            )
            questions.append(
                {"id": "outline_height_mm", "text": "外形の高さ(mm)", "type": "float", "default": height_est}
            )

    thickness = None
    for c in mml.get("constraints", []):
        if c.get("kind") == "min_thickness":
            thickness = c.get("value_mm")
            break
    if thickness is None:
        questions.append({"id": "thickness_mm", "text": "板厚は何mmですか？", "type": "float"})

    holes = mml.get("geometry", {}).get("holes", [])
    if not holes and not _is_gear(mml):
        questions.append(
            {
                "id": "hole_centers_mm",
                "text": "穴中心座標(mm) 例: 10,10; 30,10",
                "type": "text",
            }
        )
        questions.append({"id": "hole_diameter_mm", "text": "穴の直径(mm)", "type": "float"})
    elif any(h.get("diameter_mm") is None for h in holes):
        questions.append({"id": "hole_standard", "text": "穴の規格は？ (M3/M4/M5/M6/M8)", "type": "str"})
        questions.append({"id": "hole_diameter_mm", "text": "穴の直径(mm) ※規格が不明な場合", "type": "float"})
    elif holes and not _is_gear(mml):
        first_dia = holes[0].get("diameter_mm")
        questions.append(
            {"id": "hole_diameter_mm", "text": "穴の直径(mm)（確認）", "type": "float", "default": first_dia}
        )

    bend = mml.get("geometry", {}).get("bend")
    if not bend:
        questions.append(
            {
                "id": "bend_line_mm",
                "text": "曲げ線(mm) 例: 10,0; 10,50（不要なら空欄）",
                "type": "text",
            }
        )
    else:
        if bend.get("angle_deg") is None:
            questions.append({"id": "bend_angle_deg", "text": "曲げ角度は何度ですか？", "type": "float"})
        if bend.get("inner_radius_mm") is None:
            questions.append({"id": "bend_radius_mm", "text": "曲げ内Rは何mmですか？", "type": "float"})

    return questions


def _ai_generate_next_draw_question(mml, draw_answers, asked_ids, asked_texts):
    """
    具体設計フェーズでAIが動的に次の質問を生成する。

    AIがMMLの内容と現在の回答状況を見て、次に聞くべき質問を1つ生成する。
    寸法だけでなく、安全率・精度・材質・表面処理・加工方法なども含む。
    固定の質問リストではなく、文脈に応じて最適な質問を動的に決定する。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    # MMLから部品情報を抽出
    part_type = mml.get("part", "Unknown")
    intent = mml.get("intent", {})
    geometry = mml.get("geometry", {})
    constraints = mml.get("constraints", [])

    # 既に確定している情報
    known_details = {}
    for key, val in draw_answers.items():
        if val is not None:
            known_details[key] = val

    # 形状から推測できる情報
    outline = geometry.get("outline", {}).get("points_mm", [])
    holes = geometry.get("holes", [])
    bend = geometry.get("bend")

    # 質問カテゴリの例示
    question_categories = [
        "寸法（幅・高さ・厚み・直径・穴径など）",
        "材質（鉄・アルミ・ステンレス・樹脂など）",
        "安全率（1.5〜3.0など、用途に応じた値）",
        "精度・公差（±0.1mm、H7/g6など）",
        "表面処理（メッキ・塗装・アルマイト・研磨など）",
        "加工方法（レーザー切断・プレス・旋盤・フライスなど）",
        "熱処理（焼入れ・焼戻し・浸炭など）",
        "強度要件（引張強度・降伏点など）",
        "環境条件（使用温度・湿度・腐食環境など）",
    ]

    payload = {
        "part_type": part_type,
        "intent": intent,
        "outline_points_count": len(outline),
        "holes_count": len(holes),
        "has_bend": bend is not None,
        "constraints": constraints,
        "known_details": known_details,
        "asked_ids": asked_ids,
        "asked_texts": asked_texts,
        "question_categories": question_categories,
        "instruction": (
            "図面生成と製造に必要な具体的な情報を1つだけ質問してください。"
            "以下のカテゴリから、部品の用途・機能・制約条件を考慮して最も重要なものを選んでください："
            "寸法、材質、安全率、精度・公差、表面処理、加工方法、熱処理、強度要件、環境条件。"
            "既にknown_detailsにある情報は聞かないでください。"
            "asked_textsに意味的に近い質問は出さないでください。"
            "必要な情報が十分に集まったと判断した場合は {\"done\": true} を返してください。"
            "返答はJSONのみ: {\"id\": \"...\", \"text\": \"...\", \"type\": \"float|int|text\", \"category\": \"...\"} "
            "または {\"done\": true}."
        ),
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは機械設計と製造の専門家です。図面を生成し部品を製造するために必要な"
                        "具体的な情報をユーザーに質問します。寸法だけでなく、材質・安全率・精度・"
                        "表面処理・加工方法なども含めて、部品の用途や機能を考慮した適切な順序で"
                        "質問してください。日本語で質問を生成してください。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return None
        data = json.loads(text)
        if data.get("done") is True:
            return {"done": True}
        if data.get("id") and data.get("text"):
            return {
                "id": data["id"],
                "text": data["text"],
                "type": data.get("type", "text"),
                "category": data.get("category", "その他"),
            }
    except Exception:
        return None
    return None


def _ai_answer_draw_question(message, mml, draw_answers):
    """
    具体設計フェーズでユーザーからの質問に回答する。
    寸法・材質・安全率・精度・加工方法など全般に対応。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "具体的な情報を教えてください。"

    client = OpenAI(api_key=api_key)

    part_type = mml.get("part", "Unknown")
    intent = mml.get("intent", {})

    payload = {
        "part_type": part_type,
        "intent": intent,
        "known_details": {k: v for k, v in draw_answers.items() if v is not None},
        "message": message,
        "instruction": (
            "図面・製造に関する相談に短く答えてください。"
            "部品の用途や機能を考慮して、寸法・材質・安全率・精度・表面処理・加工方法など"
            "適切な目安や考え方を助言してください。"
        ),
    }

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "機械設計と製造の専門家として、図面や製造に関する相談に答えます。"
                        "寸法、材質、安全率、精度・公差、表面処理、加工方法、熱処理など"
                        "幅広い質問に対応してください。"
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "もう少し詳しく教えてください。"


def _parse_points(value):
    if not value:
        return []
    if isinstance(value, list):
        points = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    points.append([float(item[0]), float(item[1])])
                except (TypeError, ValueError):
                    continue
        return points
    text = str(value)
    points = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        points.append([x, y])
    return points


def _circle_outline(diameter_mm, segments=64):
    if diameter_mm is None:
        return []
    radius = float(diameter_mm) / 2.0
    points = []
    for i in range(segments):
        theta = 2.0 * 3.14159265 * i / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        points.append([round(x, 3), round(y, 3)])
    return points


def _rounded_rect_outline(width_mm, height_mm, radius_mm, segments=10):
    width = float(width_mm)
    height = float(height_mm)
    radius = min(float(radius_mm), width / 2.0, height / 2.0)
    if radius <= 0:
        return [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]
    points = []
    for i in range(segments + 1):
        theta = -math.pi / 2 + (math.pi / 2) * i / segments
        points.append([width - radius + radius * math.cos(theta), radius + radius * math.sin(theta)])
    for i in range(segments + 1):
        theta = 0 + (math.pi / 2) * i / segments
        points.append([width - radius + radius * math.cos(theta), height - radius + radius * math.sin(theta)])
    for i in range(segments + 1):
        theta = math.pi / 2 + (math.pi / 2) * i / segments
        points.append([radius + radius * math.cos(theta), height - radius + radius * math.sin(theta)])
    for i in range(segments + 1):
        theta = math.pi + (math.pi / 2) * i / segments
        points.append([radius + radius * math.cos(theta), radius + radius * math.sin(theta)])
    return [[round(p[0], 3), round(p[1], 3)] for p in points]


def _gear_outline(outer_diameter_mm, teeth_count):
    if outer_diameter_mm is None or teeth_count is None:
        return []
    teeth = max(6, int(teeth_count))
    outer_r = float(outer_diameter_mm) / 2.0
    root_r = outer_r * 0.85
    points = []
    total = teeth * 2
    for i in range(total):
        theta = 2.0 * 3.14159265 * i / total
        r = outer_r if i % 2 == 0 else root_r
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append([round(x, 3), round(y, 3)])
    return points


def _apply_draw_answers(mml, answers):
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    if not outline:
        if _is_gear(mml) and answers.get("outer_diameter_mm") and answers.get("teeth_count"):
            diameter = answers.get("outer_diameter_mm")
            try:
                teeth_count = int(answers.get("teeth_count"))
            except (TypeError, ValueError):
                teeth_count = None
            mml.setdefault("geometry", {})["outline"] = {
                "type": "polygon",
                "points_mm": _gear_outline(diameter, teeth_count),
            }
        elif _is_gear(mml) and answers.get("outer_diameter_mm"):
            diameter = answers.get("outer_diameter_mm")
            mml.setdefault("geometry", {})["outline"] = {
                "type": "polygon",
                "points_mm": _circle_outline(diameter),
            }
        elif _is_gear(mml) and answers.get("outer_diameter_mm") is None:
            pass
        width = answers.get("outline_width_mm")
        height = answers.get("outline_height_mm")
        if width and height and not _is_template_part(mml):
            mml.setdefault("geometry", {})["outline"] = {
                "type": "polygon",
                "points_mm": [[0, 0], [width, 0], [width, height], [0, height]],
            }

    thickness = answers.get("thickness_mm")
    if thickness is not None:
        found = False
        for c in mml.get("constraints", []):
            if c.get("kind") == "min_thickness":
                c["value_mm"] = thickness
                found = True
                break
        if not found:
            mml.setdefault("constraints", []).append({"kind": "min_thickness", "value_mm": thickness})

    hole_standard = answers.get("hole_standard")
    hole_diameter_mm = answers.get("hole_diameter_mm")
    centers_text = answers.get("hole_centers_mm")
    if centers_text:
        centers = _parse_points(centers_text)
        mml.setdefault("geometry", {})["holes"] = [
            {"type": "clearance", "standard": "custom", "center_mm": c, "diameter_mm": None}
            for c in centers
        ]
    if _is_gear(mml) and answers.get("bore_diameter_mm") is not None:
        bore = float(answers.get("bore_diameter_mm"))
        mml.setdefault("geometry", {})["holes"] = [
            {"type": "clearance", "standard": "custom", "center_mm": [0.0, 0.0], "diameter_mm": bore}
        ]

    if hole_standard:
        hole_standard = hole_standard.strip().upper()
        hole_diameter_mm = HOLE_CLEARANCE_MM.get(hole_standard, hole_diameter_mm)
    if hole_diameter_mm is not None:
        for h in mml.get("geometry", {}).get("holes", []):
            h["diameter_mm"] = hole_diameter_mm
            h["standard"] = hole_standard or "custom"

    bend = mml.get("geometry", {}).get("bend")
    bend_line_text = answers.get("bend_line_mm")
    if not bend and bend_line_text:
        line_points = _parse_points(bend_line_text)
        if len(line_points) == 2:
            mml.setdefault("geometry", {})["bend"] = {
                "line_mm": line_points,
                "angle_deg": None,
                "inner_radius_mm": None,
            }
            bend = mml.get("geometry", {}).get("bend")

    if bend:
        if answers.get("bend_angle_deg") is not None:
            bend["angle_deg"] = answers.get("bend_angle_deg")
        if answers.get("bend_radius_mm") is not None:
            bend["inner_radius_mm"] = answers.get("bend_radius_mm")

    # 材質・安全率・精度・表面処理・加工方法などの追加情報をMMLに反映
    manufacturing = mml.setdefault("manufacturing", {})

    # 材質
    if answers.get("material"):
        manufacturing["material"] = answers.get("material")
    if answers.get("material_grade"):
        manufacturing["material_grade"] = answers.get("material_grade")

    # 安全率
    if answers.get("safety_factor") is not None:
        manufacturing["safety_factor"] = answers.get("safety_factor")

    # 精度・公差
    if answers.get("tolerance"):
        manufacturing["tolerance"] = answers.get("tolerance")
    if answers.get("surface_roughness"):
        manufacturing["surface_roughness"] = answers.get("surface_roughness")

    # 表面処理
    if answers.get("surface_treatment"):
        manufacturing["surface_treatment"] = answers.get("surface_treatment")

    # 加工方法
    if answers.get("machining_method"):
        manufacturing["machining_method"] = answers.get("machining_method")

    # 熱処理
    if answers.get("heat_treatment"):
        manufacturing["heat_treatment"] = answers.get("heat_treatment")

    # 強度要件
    if answers.get("tensile_strength"):
        manufacturing["tensile_strength"] = answers.get("tensile_strength")
    if answers.get("yield_strength"):
        manufacturing["yield_strength"] = answers.get("yield_strength")
    if answers.get("hardness"):
        manufacturing["hardness"] = answers.get("hardness")

    # 環境条件
    if answers.get("operating_temperature"):
        manufacturing["operating_temperature"] = answers.get("operating_temperature")
    if answers.get("environment"):
        manufacturing["environment"] = answers.get("environment")

    # その他の動的に追加された情報
    known_keys = {
        "outline_width_mm", "outline_height_mm", "outer_diameter_mm", "bore_diameter_mm",
        "teeth_count", "thickness_mm", "hole_centers_mm", "hole_standard", "hole_diameter_mm",
        "bend_line_mm", "bend_angle_deg", "bend_radius_mm", "material", "material_grade",
        "safety_factor", "tolerance", "surface_roughness", "surface_treatment",
        "machining_method", "heat_treatment", "tensile_strength", "yield_strength",
        "hardness", "operating_temperature", "environment",
    }
    for key, val in answers.items():
        if key not in known_keys and val is not None:
            manufacturing[key] = val

    # 空のmanufacturingは削除
    if not manufacturing:
        mml.pop("manufacturing", None)

    return mml


def _collect_missing_draw(mml, answers):
    missing = {}
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    if not outline:
        if _is_gear(mml):
            if answers.get("outer_diameter_mm") is None:
                missing["outer_diameter_mm"] = None
            if answers.get("bore_diameter_mm") is None:
                missing["bore_diameter_mm"] = None
            if answers.get("teeth_count") is None:
                missing["teeth_count"] = None
        elif not _is_template_part(mml):
            if answers.get("outline_width_mm") is None:
                missing["outline_width_mm"] = None
            if answers.get("outline_height_mm") is None:
                missing["outline_height_mm"] = None

    thickness = None
    for c in mml.get("constraints", []):
        if c.get("kind") == "min_thickness":
            thickness = c.get("value_mm")
            break
    if thickness is None and answers.get("thickness_mm") is None:
        missing["thickness_mm"] = None

    holes = mml.get("geometry", {}).get("holes", [])
    if not holes and not _is_gear(mml):
        if answers.get("hole_centers_mm") is None:
            missing["hole_centers_mm"] = None
        if answers.get("hole_diameter_mm") is None:
            missing["hole_diameter_mm"] = None
    elif any(h.get("diameter_mm") is None for h in holes):
        if answers.get("hole_diameter_mm") is None and answers.get("hole_standard") is None:
            missing["hole_standard"] = None
            missing["hole_diameter_mm"] = None

    bend = mml.get("geometry", {}).get("bend")
    if not bend:
        if answers.get("bend_line_mm") is None:
            missing["bend_line_mm"] = None
    else:
        if bend.get("angle_deg") is None and answers.get("bend_angle_deg") is None:
            missing["bend_angle_deg"] = None
        if bend.get("inner_radius_mm") is None and answers.get("bend_radius_mm") is None:
            missing["bend_radius_mm"] = None

    return missing


def _auto_fill_drawing(mml, missing, note=None):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return {}
    client = OpenAI(api_key=api_key)
    payload = {
        "missing": list(missing.keys()),
        "candidates": [
            "outline_width_mm",
            "outline_height_mm",
            "outer_diameter_mm",
            "bore_diameter_mm",
            "teeth_count",
            "thickness_mm",
            "hole_centers_mm",
            "hole_diameter_mm",
            "hole_standard",
            "bend_line_mm",
            "bend_angle_deg",
            "bend_radius_mm",
        ],
        "mml": mml,
        "part": mml.get("part"),
        "intent": mml.get("intent"),
        "notes": (
            "Pick which fields from candidates should be filled based on part and intent. "
            "Return a JSON object with only the keys you choose to fill. "
            "Values must be numbers or strings as appropriate."
        ),
        "user_note": note,
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You fill missing drawing parameters with plausible defaults."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        if not text:
            return {}
        data = json.loads(text)
        return {k: data.get(k) for k in missing.keys()}
    except Exception:
        return {}


def _has_geometry(mml):
    geom = mml.get("geometry", {}) or {}
    outline = geom.get("outline", {}).get("points_mm", []) or []
    holes = geom.get("holes", []) or []
    bend = geom.get("bend")
    return bool(outline) or bool(holes) or bool(bend)


def _make_placeholder_geometry(mml):
    part = (mml.get("part") or "").lower()
    geom = mml.setdefault("geometry", {})
    if geom.get("outline", {}).get("points_mm"):
        return mml
    intent = mml.get("intent", {}) or {}
    arm_cfg = intent.get("arm_config") or {}
    arm_dims = intent.get("arm_dims") or {}
    try:
        reach = float(arm_cfg.get("reach_mm", 300.0))
    except (TypeError, ValueError):
        reach = 300.0
    scale = max(0.6, min(1.4, reach / 300.0))

    if "shaft" in part:
        diameter = float(arm_dims.get("shaft_diameter_mm", 8.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(diameter, segments=96)}
        return mml
    if "link" in part or "arm" in part:
        length = float(arm_dims.get("link_length_mm", 140.0 * scale))
        width = float(arm_dims.get("link_width_mm", 26.0 * scale))
        fillet = float(arm_dims.get("link_fillet_mm", 5.0 * scale))
        offset = float(arm_dims.get("link_hole_offset_mm", 16.0 * scale))
        hole_d = float(arm_dims.get("joint_hole_diameter_mm", 6.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _rounded_rect_outline(length, width, fillet, segments=12)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [offset, width / 2.0],
                "diameter_mm": hole_d,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [length - offset, width / 2.0],
                "diameter_mm": hole_d,
            },
        ]
        return mml
    if "joint" in part:
        joint_od = float(arm_dims.get("joint_outer_diameter_mm", 36.0 * scale))
        hole_d = float(arm_dims.get("joint_hole_diameter_mm", 8.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(joint_od, segments=120)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [0.0, 0.0],
                "diameter_mm": hole_d,
            }
        ]
        return mml
    if "base" in part:
        base_w = float(arm_dims.get("base_width_mm", 100.0 * scale))
        base_h = float(arm_dims.get("base_height_mm", 70.0 * scale))
        fillet = float(arm_dims.get("base_fillet_mm", 6.0 * scale))
        hole_d = float(arm_dims.get("joint_hole_diameter_mm", 6.0 * scale))
        offset = float(arm_dims.get("base_hole_offset_mm", 12.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _rounded_rect_outline(base_w, base_h, fillet, segments=12)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [offset, offset],
                "diameter_mm": hole_d,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [base_w - offset, offset],
                "diameter_mm": hole_d,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [base_w - offset, base_h - offset],
                "diameter_mm": hole_d,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [offset, base_h - offset],
                "diameter_mm": hole_d,
            },
        ]
        return mml
    if "end_effector" in part or "gripper" in part:
        geom["outline"] = {"type": "polygon", "points_mm": [[0, 0], [40, 0], [40, 20], [0, 20]]}
        return mml
    if "actuator" in part or "motor" in part or "servo" in part:
        motor_od = float(arm_dims.get("motor_outer_diameter_mm", 40.0 * scale))
        shaft_d = float(arm_dims.get("shaft_diameter_mm", 6.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(motor_od, segments=120)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [0.0, 0.0],
                "diameter_mm": shaft_d,
            }
        ]
        return mml
    if "motor_mount" in part or "mount" in part:
        mount_w = float(arm_dims.get("motor_mount_width_mm", 70.0 * scale))
        mount_h = float(arm_dims.get("motor_mount_height_mm", 50.0 * scale))
        fillet = float(arm_dims.get("motor_mount_fillet_mm", 5.0 * scale))
        shaft_d = float(arm_dims.get("shaft_diameter_mm", 6.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _rounded_rect_outline(mount_w, mount_h, fillet, segments=12)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [10.0 * scale, 10.0 * scale],
                "diameter_mm": 5.0 * scale,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [mount_w - 10.0 * scale, 10.0 * scale],
                "diameter_mm": 5.0 * scale,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [mount_w - 10.0 * scale, mount_h - 10.0 * scale],
                "diameter_mm": 5.0 * scale,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [10.0 * scale, mount_h - 10.0 * scale],
                "diameter_mm": 5.0 * scale,
            },
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [mount_w / 2.0, mount_h / 2.0],
                "diameter_mm": shaft_d,
            },
        ]
        return mml
    if "bracket" in part:
        geom["outline"] = {
            "type": "polygon",
            "points_mm": [[0, 0], [60, 0], [60, 15], [20, 15], [20, 50], [0, 50]],
        }
        geom["holes"] = [
            {"type": "clearance", "standard": "custom", "center_mm": [10.0, 10.0], "diameter_mm": 6.0},
            {"type": "clearance", "standard": "custom", "center_mm": [10.0, 40.0], "diameter_mm": 6.0},
        ]
        return mml
    if "bearing" in part:
        od = float(arm_dims.get("bearing_outer_diameter_mm", 30.0 * scale))
        id_d = float(arm_dims.get("bearing_inner_diameter_mm", 12.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(od, segments=120)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [0.0, 0.0],
                "diameter_mm": id_d,
            }
        ]
        return mml
    if "spacer" in part:
        od = float(arm_dims.get("spacer_outer_diameter_mm", 22.0 * scale))
        id_d = float(arm_dims.get("shaft_diameter_mm", 6.0 * scale))
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(od, segments=96)}
        geom["holes"] = [
            {
                "type": "clearance",
                "standard": "custom",
                "center_mm": [0.0, 0.0],
                "diameter_mm": id_d,
            }
        ]
        return mml
    if "rotor" in part or "stator" in part:
        geom["outline"] = {"type": "spline", "points_mm": _circle_outline(40.0, segments=120)}
        return mml
    if "housing" in part or "case" in part:
        geom["outline"] = {"type": "polygon", "points_mm": [[0, 0], [60, 0], [60, 40], [0, 40]]}
        return mml
    if "gear" in part:
        geom["outline"] = {"type": "polygon", "points_mm": _gear_outline(60.0, 24)}
        geom["holes"] = [
            {"type": "clearance", "standard": "custom", "center_mm": [0.0, 0.0], "diameter_mm": 12.0}
        ]
        return mml

    # 汎用フォールバック
    geom["outline"] = {"type": "polygon", "points_mm": [[0, 0], [50, 0], [50, 30], [0, 30]]}
    return mml


def _infer_subcomponents(intent, inferred_part):
    if not inferred_part:
        return []
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        if inferred_part.lower() == "motor":
            return ["housing", "shaft", "rotor", "end_cap", "stator"]
        if inferred_part.lower() == "robotarm":
            return ["base", "joint_1", "link_1", "joint_2", "link_2", "end_effector"]
        return []
    client = OpenAI(api_key=api_key)
    payload = {
        "part": inferred_part,
        "intent": intent,
        "instruction": (
            "Return JSON with a list of subcomponents for the part. "
            "Use simple names like Base, Joint, Link, EndEffector, Actuator, Shaft, Gear, Housing. "
            "Avoid control/electrical items."
        ),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You propose subcomponents for mechanical parts."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
        data = json.loads(text)
        items = data.get("subcomponents")
        if isinstance(items, list):
            return [str(x) for x in items if str(x).strip()]
    except Exception:
        pass
    return []


@app.route("/draw", methods=["GET"])
def draw_start():
    run_id = request.args.get("run_id")
    return render_template("draw_start.html", run_id=run_id)


@app.route("/draw/load", methods=["POST"])
def draw_load():
    run_id = request.form.get("run_id") or None
    mml = None
    run_dir = None

    upload = request.files.get("mml_file")
    if upload and upload.filename:
        filename = secure_filename(upload.filename)
        _, ext = os.path.splitext(filename)
        if ext.lower() != ".json":
            return render_template("draw_start.html", error="mml.json をアップロードしてください。")
        run_id = new_run_id()
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        ensure_dir(run_dir)
        mml_path = os.path.join(run_dir, "mml.json")
        upload.save(mml_path)
        mml = read_json(mml_path)
    elif run_id:
        run_dir = os.path.join(OUTPUT_ROOT, run_id)
        mml_path = os.path.join(run_dir, "mml.json")
        if not os.path.exists(mml_path):
            return render_template("draw_start.html", error="mml.json が見つかりません。")
        mml = read_json(mml_path)
    else:
        return render_template("draw_start.html", error="mml.json のアップロードか run id の入力が必要です。")

    if mml is not None and mml.get("intent") is not None:
        normalized = _normalize_subcomponents(mml["intent"].get("subcomponents"))
        if normalized:
            normalized = _ensure_robotarm_components(normalized, mml["intent"].get("inferred_part"))
            mml["intent"]["subcomponents"] = normalized
        if (mml.get("part") in {None, "", "Unknown"}) and mml["intent"].get("inferred_part"):
            mml["part"] = mml["intent"]["inferred_part"]

    # 対話形式で寸法を確認するための状態初期化
    draw_answers = {}
    asked_ids = []
    asked_texts = []
    chat_history = []

    # AIが最初の質問を動的に生成
    first_question = _ai_generate_next_draw_question(mml, draw_answers, asked_ids, asked_texts)
    if first_question is None or first_question.get("done"):
        # AIが質問不要と判断した場合、従来の固定質問にフォールバック
        questions = _draw_questions(mml)
        if questions:
            first_question = questions[0]
        else:
            first_question = None

    if first_question and first_question.get("id"):
        asked_ids.append(first_question["id"])
    if first_question and first_question.get("text"):
        asked_texts.append(first_question["text"])

    write_json(os.path.join(run_dir, "mml.json"), mml)
    api_error = not os.getenv("OPENAI_API_KEY")
    return render_template(
        "draw_chat.html",
        run_id=run_id,
        current_question=first_question,
        chat_history=chat_history,
        draw_answers_json=json.dumps(draw_answers, ensure_ascii=False),
        asked_ids_json=json.dumps(asked_ids, ensure_ascii=False),
        asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
        mml_preview=json.dumps(mml, indent=2),
        api_error=api_error,
    )


@app.route("/draw/load", methods=["GET"])
def draw_load_get():
    return redirect(url_for("draw_start"))


@app.route("/draw/chat", methods=["POST"])
def draw_chat():
    """
    具体寸法フェーズの対話処理。
    AIが動的に質問を生成し、ユーザーの回答を収集する。
    """
    run_id = request.form.get("run_id")
    if not run_id:
        return render_template("draw_start.html", error="run id がありません。")

    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    mml_path = os.path.join(run_dir, "mml.json")
    if not os.path.exists(mml_path):
        return render_template("draw_start.html", error="mml.json が見つかりません。")

    mml = read_json(mml_path)

    # フォームから状態を復元
    draw_answers_json = request.form.get("draw_answers_json", "{}")
    asked_ids_json = request.form.get("asked_ids_json", "[]")
    asked_texts_json = request.form.get("asked_texts_json", "[]")
    chat_history_json = request.form.get("chat_history_json", "[]")

    try:
        draw_answers = json.loads(draw_answers_json)
    except Exception:
        draw_answers = {}
    try:
        asked_ids = json.loads(asked_ids_json)
    except Exception:
        asked_ids = []
    try:
        asked_texts = json.loads(asked_texts_json)
    except Exception:
        asked_texts = []
    try:
        chat_history = json.loads(chat_history_json)
    except Exception:
        chat_history = []

    # ユーザーの回答を取得
    current_id = request.form.get("current_id")
    current_text = request.form.get("current_text")
    current_type = request.form.get("current_type", "text")
    user_input = request.form.get("user_input", "").strip()
    chat_action = request.form.get("chat_action", "answer")

    # ユーザーが質問した場合
    if chat_action == "question":
        if user_input:
            chat_history.append({"role": "user", "text": user_input})
            reply = _ai_answer_draw_question(user_input, mml, draw_answers)
            chat_history.append({"role": "bot", "text": reply})

        # 現在の質問を維持
        current_question = None
        if current_id:
            current_question = {"id": current_id, "text": current_text, "type": current_type or "text"}

        return render_template(
            "draw_chat.html",
            run_id=run_id,
            current_question=current_question,
            chat_history=chat_history,
            draw_answers_json=json.dumps(draw_answers, ensure_ascii=False),
            asked_ids_json=json.dumps(asked_ids, ensure_ascii=False),
            asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
            mml_preview=json.dumps(mml, indent=2),
        )

    # ユーザーの回答を処理
    if user_input and current_id:
        # 型に応じて値を変換
        if current_type == "float":
            try:
                draw_answers[current_id] = float(user_input)
            except ValueError:
                draw_answers[current_id] = user_input
        elif current_type == "int":
            try:
                draw_answers[current_id] = int(user_input)
            except ValueError:
                draw_answers[current_id] = user_input
        else:
            draw_answers[current_id] = user_input

        # チャット履歴に追加
        if current_text:
            chat_history.append({"role": "bot", "text": current_text})
        chat_history.append({"role": "user", "text": user_input})

    # 完了ボタンが押された場合
    if chat_action == "finish":
        # 図面生成ページへ
        return render_template(
            "draw_chat.html",
            run_id=run_id,
            current_question=None,
            chat_history=chat_history,
            draw_answers_json=json.dumps(draw_answers, ensure_ascii=False),
            asked_ids_json=json.dumps(asked_ids, ensure_ascii=False),
            asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
            mml_preview=json.dumps(mml, indent=2),
            ready_to_generate=True,
        )

    # AIが次の質問を動的に生成
    next_q = _ai_generate_next_draw_question(mml, draw_answers, asked_ids, asked_texts)

    if next_q is None or next_q.get("done"):
        # 質問が完了した場合、図面生成ページへ
        return render_template(
            "draw_chat.html",
            run_id=run_id,
            current_question=None,
            chat_history=chat_history,
            draw_answers_json=json.dumps(draw_answers, ensure_ascii=False),
            asked_ids_json=json.dumps(asked_ids, ensure_ascii=False),
            asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
            mml_preview=json.dumps(mml, indent=2),
            ready_to_generate=True,
        )

    # 次の質問を追加
    if next_q.get("id"):
        asked_ids.append(next_q["id"])
    if next_q.get("text"):
        asked_texts.append(next_q["text"])

    return render_template(
        "draw_chat.html",
        run_id=run_id,
        current_question=next_q,
        chat_history=chat_history,
        draw_answers_json=json.dumps(draw_answers, ensure_ascii=False),
        asked_ids_json=json.dumps(asked_ids, ensure_ascii=False),
        asked_texts_json=json.dumps(asked_texts, ensure_ascii=False),
        mml_preview=json.dumps(mml, indent=2),
    )


@app.route("/draw/generate", methods=["POST"])
def draw_generate():
    """
    対話で収集した寸法情報を使って図面を生成する。
    サブコンポーネントがある場合はマルチコンポーネント生成を行う。
    """
    run_id = request.form.get("run_id")
    if not run_id:
        return render_template("draw_start.html", error="run id がありません。")

    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    mml_path = os.path.join(run_dir, "mml.json")
    if not os.path.exists(mml_path):
        return render_template("draw_start.html", error="mml.json が見つかりません。")

    mml = read_json(mml_path)

    # 対話で収集した回答を取得
    draw_answers_json = request.form.get("draw_answers_json", "{}")
    try:
        draw_answers = json.loads(draw_answers_json)
    except Exception:
        draw_answers = {}

    # サブコンポーネント正規化
    if mml.get("intent") is not None:
        normalized = _normalize_subcomponents(mml["intent"].get("subcomponents"))
        if normalized:
            normalized = _ensure_robotarm_components(normalized, mml["intent"].get("inferred_part"))
            mml["intent"]["subcomponents"] = normalized
        if (mml.get("part") in {None, "", "Unknown"}) and mml["intent"].get("inferred_part"):
            mml["part"] = mml["intent"]["inferred_part"]

    def _fill_and_draw_component(target_mml, prefix):
        """コンポーネント1つ分の図面・STL生成。"""
        updated = _apply_draw_answers(target_mml, draw_answers)
        missing = _collect_missing_draw(updated, draw_answers)
        if missing:
            filled = _auto_fill_drawing(updated, missing, note=None)
            if not filled:
                filled = {}
            if "outer_diameter_mm" in missing and "outer_diameter_mm" not in filled:
                filled["outer_diameter_mm"] = 100.0
            if "bore_diameter_mm" in missing and "bore_diameter_mm" not in filled:
                filled["bore_diameter_mm"] = 20.0
            if "teeth_count" in missing and "teeth_count" not in filled:
                filled["teeth_count"] = 24
            if "outline_width_mm" in missing and "outline_width_mm" not in filled:
                filled["outline_width_mm"] = 100.0
            if "outline_height_mm" in missing and "outline_height_mm" not in filled:
                filled["outline_height_mm"] = 60.0
            if "thickness_mm" in missing and "thickness_mm" not in filled:
                filled["thickness_mm"] = 2.0
            if "hole_diameter_mm" in missing and "hole_diameter_mm" not in filled:
                filled["hole_diameter_mm"] = 5.0
            if "hole_centers_mm" in missing and "hole_centers_mm" not in filled:
                filled["hole_centers_mm"] = ""
            if filled:
                updated = _apply_draw_answers(updated, filled)

        if not _has_geometry(updated):
            updated = _make_placeholder_geometry(updated)

        mml_name = f"{prefix}mml.json"
        dxf_name = f"{prefix}drawing.dxf"
        png_name = f"{prefix}drawing.png"
        stl_name = f"{prefix}model.stl"
        report_name = f"{prefix}drawing_report.json"

        write_json(os.path.join(run_dir, mml_name), updated)
        draw_dxf(updated, os.path.join(run_dir, dxf_name))
        draw_png(updated, os.path.join(run_dir, png_name))
        try:
            write_stl(updated, os.path.join(run_dir, stl_name))
        except Exception:
            pass
        write_json(os.path.join(run_dir, report_name), {"answers": draw_answers})
        return {
            "mml": mml_name,
            "dxf": dxf_name,
            "png": png_name,
            "stl": stl_name,
            "drawing_report": report_name,
        }

    # サブコンポーネントがある場合はマルチコンポーネント生成
    subcomponents = (mml.get("intent") or {}).get("subcomponents") or []
    if isinstance(subcomponents, list) and len(subcomponents) > 1:
        outputs_multi = []
        for idx, name in enumerate(subcomponents, start=1):
            comp = copy.deepcopy(mml)
            comp["part"] = str(name)
            comp.setdefault("intent", {})["subcomponent"] = str(name)
            # コンポーネント固有のジオメトリを生成するため既存ジオメトリをクリア
            comp["geometry"] = {}
            prefix = f"comp{idx}_"
            files = _fill_and_draw_component(comp, prefix)
            if files:
                outputs_multi.append({"name": str(name), "files": files})

        # メイン MML も保存
        write_json(os.path.join(run_dir, "mml.json"), mml)

        if not outputs_multi:
            return render_template(
                "draw_result.html",
                run_id=run_id,
                outputs=None,
                outputs_multi=None,
            )
        assembly_stl = _assemble_stl(run_dir, outputs_multi)
        return render_template(
            "draw_result.html",
            run_id=run_id,
            outputs_multi=outputs_multi,
            assembly_stl=assembly_stl,
        )

    # 単一コンポーネントの場合
    outputs = _fill_and_draw_component(mml, "")
    if outputs is None:
        return render_template(
            "draw_result.html",
            run_id=run_id,
            outputs=None,
            outputs_multi=None,
        )
    return render_template(
        "draw_result.html",
        run_id=run_id,
        outputs=outputs,
    )


@app.route("/draw/run", methods=["POST"])
def draw_run():
    run_id = request.form.get("run_id")
    if not run_id:
        return render_template("draw_start.html", error="run id がありません。")
    run_dir = os.path.join(OUTPUT_ROOT, run_id)
    mml_path = os.path.join(run_dir, "mml.json")
    if not os.path.exists(mml_path):
        return render_template("draw_start.html", error="mml.json が見つかりません。")
    mml = read_json(mml_path)
    if mml.get("intent") is not None:
        normalized = _normalize_subcomponents(mml["intent"].get("subcomponents"))
        if normalized:
            normalized = _ensure_robotarm_components(normalized, mml["intent"].get("inferred_part"))
            mml["intent"]["subcomponents"] = normalized
        if (mml.get("part") in {None, "", "Unknown"}) and mml["intent"].get("inferred_part"):
            mml["part"] = mml["intent"]["inferred_part"]

    answers = {
        "outline_width_mm": _get_form_float("outline_width_mm"),
        "outline_height_mm": _get_form_float("outline_height_mm"),
        "outer_diameter_mm": _get_form_float("outer_diameter_mm"),
        "bore_diameter_mm": _get_form_float("bore_diameter_mm"),
        "teeth_count": _get_form_int("teeth_count"),
        "thickness_mm": _get_form_float("thickness_mm"),
        "hole_centers_mm": _get_form_str("hole_centers_mm"),
        "hole_standard": _get_form_str("hole_standard"),
        "hole_diameter_mm": _get_form_float("hole_diameter_mm"),
        "bend_line_mm": _get_form_str("bend_line_mm"),
        "bend_angle_deg": _get_form_float("bend_angle_deg"),
        "bend_radius_mm": _get_form_float("bend_radius_mm"),
    }
    action = request.form.get("action", "generate")
    advice_note = _get_form_str("advice_note")
    mml = _apply_draw_answers(mml, answers)
    if action == "suggest":
        missing = _collect_missing_draw(mml, answers)
        suggestions = {}
        if missing:
            suggestions = _auto_fill_drawing(mml, missing, note=advice_note)
        questions = _draw_questions(mml)
        return render_template(
            "draw_chat.html",
            run_id=run_id,
            questions=questions,
            mml_preview=json.dumps(mml, indent=2),
            suggestions=suggestions,
            advice_note=advice_note or "",
            notice="AIの提案を反映しました。必要なら修正してから生成してください。",
        )
    def _fill_and_draw(target_mml, prefix):
        updated = _apply_draw_answers(target_mml, answers)
        missing = _collect_missing_draw(updated, answers)
        if missing:
            filled = _auto_fill_drawing(updated, missing, note=advice_note)
            if not filled:
                # AIが使えない場合のフォールバック既定値。
                if "outer_diameter_mm" in missing:
                    filled["outer_diameter_mm"] = 100.0
                if "bore_diameter_mm" in missing:
                    filled["bore_diameter_mm"] = 20.0
                if "teeth_count" in missing:
                    filled["teeth_count"] = 24
                if "outline_width_mm" in missing:
                    filled["outline_width_mm"] = 100.0
                if "outline_height_mm" in missing:
                    filled["outline_height_mm"] = 60.0
                if "thickness_mm" in missing:
                    filled["thickness_mm"] = 2.0
                if "hole_diameter_mm" in missing:
                    filled["hole_diameter_mm"] = 5.0
                if "hole_centers_mm" in missing:
                    filled["hole_centers_mm"] = ""
            updated = _apply_draw_answers(updated, filled)

        if not _has_geometry(updated):
            updated = _make_placeholder_geometry(updated)

        mml_name = f"{prefix}mml.json"
        dxf_name = f"{prefix}drawing.dxf"
        png_name = f"{prefix}drawing.png"
        report_name = f"{prefix}drawing_report.json"
        write_json(os.path.join(run_dir, mml_name), updated)
        draw_dxf(updated, os.path.join(run_dir, dxf_name))
        draw_png(updated, os.path.join(run_dir, png_name))
        stl_name = f"{prefix}model.stl"
        write_stl(updated, os.path.join(run_dir, stl_name))
        write_json(os.path.join(run_dir, report_name), {"answers": answers})
        return {
            "mml": mml_name,
            "dxf": dxf_name,
            "png": png_name,
            "stl": stl_name,
            "drawing_report": report_name,
        }

    subcomponents = (mml.get("intent") or {}).get("subcomponents") or []
    if isinstance(subcomponents, list) and len(subcomponents) > 1:
        outputs_multi = []
        for idx, name in enumerate(subcomponents, start=1):
            comp = copy.deepcopy(mml)
            comp["part"] = str(name)
            comp.setdefault("intent", {})["subcomponent"] = str(name)
            prefix = f"comp{idx}_"
            files = _fill_and_draw(comp, prefix)
            if files:
                outputs_multi.append({"name": str(name), "files": files})
        if not outputs_multi:
            return render_template(
                "draw_result.html",
                run_id=run_id,
                outputs=None,
                outputs_multi=None,
            )
        assembly_stl = _assemble_stl(run_dir, outputs_multi)
        return render_template(
            "draw_result.html",
            run_id=run_id,
            outputs_multi=outputs_multi,
            assembly_stl=assembly_stl,
        )

    outputs = _fill_and_draw(mml, "")
    if outputs is None:
        return render_template(
            "draw_result.html",
            run_id=run_id,
            outputs=None,
            outputs_multi=None,
        )
    return render_template(
        "draw_result.html",
        run_id=run_id,
        outputs=outputs,
    )


@app.route("/draw/run", methods=["GET"])
def draw_run_get():
    return redirect(url_for("draw_start"))


if __name__ == "__main__":
    ensure_dir(OUTPUT_ROOT)
    app.run(debug=True)
