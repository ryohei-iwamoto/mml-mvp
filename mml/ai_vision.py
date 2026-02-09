import base64
import json
import os

from openai import OpenAI


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def run_ai_vision(image_path, api_key, model=None):
    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    image_b64 = _encode_image(image_path)

    schema = {
        "type": "object",
        "properties": {
            "part_hint": {"type": "string"},
            "part_hint_confidence": {"type": "number"},
            "outline": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "points_px": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    },
                },
                "required": ["type", "points_px"],
            },
            "holes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "center_px": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "radius_px": {"type": "number"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["center_px", "radius_px", "confidence"],
                },
            },
            "bend_lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "line_px": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["line_px", "confidence"],
                },
            },
            "notes_regions": {"type": "array"},
        },
        "required": ["part_hint", "part_hint_confidence", "outline", "holes", "bend_lines", "notes_regions"],
    }

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": "You extract geometric features from a line drawing and return strict JSON.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                            "Analyze the image and return JSON only. "
                            "The image may be a photo or a line drawing. "
                            "Always set part_hint to Gear/Bracket/Plate/Motor/RobotArm/Unknown and part_hint_confidence (0-1). "
                            "If you cannot extract geometry from a photo, still set part_hint and leave geometry empty. "
                            "Outline is the outer contour of the part when available. "
                            "If the outline is curved, set outline.type to spline and sample enough points for a smooth curve (>=32). "
                            "If mostly straight edges, set outline.type to polygon. "
                            "Holes are circular features. "
                            "Bend lines are long straight lines inside the outline. "
                            "Use pixel coordinates. Confidence is 0-1."
                            ),
                        },
                        {"type": "input_image", "image_base64": image_b64},
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": {"name": "vision", "schema": schema}},
        )
        text = response.output_text.strip()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract geometric features from a line drawing and return strict JSON.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                            "Analyze the image and return JSON only. "
                            "The image may be a photo or a line drawing. "
                            "Always set part_hint to Gear/Bracket/Plate/Motor/RobotArm/Unknown and part_hint_confidence (0-1). "
                            "If you cannot extract geometry from a photo, still set part_hint and leave geometry empty. "
                            "Outline is the outer contour of the part when available. "
                            "If the outline is curved, set outline.type to spline and sample enough points for a smooth curve (>=32). "
                            "If mostly straight edges, set outline.type to polygon. "
                            "Holes are circular features. "
                            "Bend lines are long straight lines inside the outline. "
                            "Use pixel coordinates. Confidence is 0-1."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
    if not text:
        raise ValueError("AI vision returned empty output")
    data = json.loads(text)
    if "part_hint" not in data:
        data["part_hint"] = "Unknown"
    if "part_hint_confidence" not in data:
        data["part_hint_confidence"] = 0.2
    return data
