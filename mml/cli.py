import argparse
import os
import sys

from dotenv import load_dotenv

from .ai_vision import run_ai_vision
from .draw import draw_dxf, draw_png
from .stl import write_stl
from .emit import emit_mml
from .pipeline import run_pipeline, run_library_pipeline
from .utils import ensure_dir, read_json, write_json
from .vision import run_vision

load_dotenv()


def _prompt_value(question):
    text = question["text"]
    qtype = question.get("type")
    while True:
        value = input(f"{text} ").strip()
        if value == "":
            return None
        if qtype == "float":
            try:
                return float(value)
            except ValueError:
                print("Please enter a number.")
        elif qtype == "bool":
            if value.lower() in {"y", "yes", "true", "1"}:
                return True
            if value.lower() in {"n", "no", "false", "0"}:
                return False
            print("Please answer y/n.")
        else:
            return value


def _load_input(input_path, use_ai=False):
    if input_path.lower().endswith(".json"):
        return read_json(input_path)
    if use_ai:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY が未設定です。 .env を作成してください。")
        model = os.getenv("OPENAI_MODEL")
        return run_ai_vision(input_path, api_key=api_key, model=model)
    return run_vision(input_path)


def cmd_vision(args):
    vision = _load_input(args.input, use_ai=args.ai)
    ensure_dir(args.output)
    out_path = os.path.join(args.output, "vision.json")
    write_json(out_path, vision)
    print(out_path)


def cmd_interact(args):
    vision = _load_input(args.input, use_ai=args.ai)
    ensure_dir(args.output)
    params = {
        "plate_width_mm": args.plate_width_mm,
        "hole_standard": args.hole_standard,
        "hole_diameter_mm": args.hole_diameter_mm,
        "thickness_mm": args.thickness_mm,
        "bend_angle_deg": args.bend_angle_deg,
        "bend_radius_mm": args.bend_radius_mm,
        "unify_holes": args.unify_holes,
    }
    prompt_fn = _prompt_value if args.chat == "rule" else None
    mml, report = emit_mml(vision, params, os.path.basename(args.input), prompt_fn=prompt_fn)
    mml_path = os.path.join(args.output, "mml.json")
    report_path = os.path.join(args.output, "report.json")
    write_json(mml_path, mml)
    write_json(report_path, report)
    print(mml_path)
    print(report_path)


def cmd_draw(args):
    mml = read_json(args.input)
    ensure_dir(args.output)
    out_path = os.path.join(args.output, "drawing.dxf")
    draw_dxf(mml, out_path)
    png_path = os.path.join(args.output, "drawing.png")
    draw_png(mml, png_path)
    stl_path = os.path.join(args.output, "model.stl")
    write_stl(mml, stl_path)
    print(out_path)
    print(png_path)
    print(stl_path)


def cmd_pipeline(args):
    params = {
        "plate_width_mm": args.plate_width_mm,
        "hole_standard": args.hole_standard,
        "hole_diameter_mm": args.hole_diameter_mm,
        "thickness_mm": args.thickness_mm,
        "bend_angle_deg": args.bend_angle_deg,
        "bend_radius_mm": args.bend_radius_mm,
        "unify_holes": args.unify_holes,
        "use_ai": args.ai,
    }
    prompt_fn = _prompt_value if args.chat == "rule" else None
    ensure_dir(args.output)
    if prompt_fn:
        vision = _load_input(args.input, use_ai=args.ai)
        vision_path = os.path.join(args.output, "vision.json")
        write_json(vision_path, vision)
        mml, report = emit_mml(vision, params, os.path.basename(args.input), prompt_fn=prompt_fn)
        mml_path = os.path.join(args.output, "mml.json")
        report_path = os.path.join(args.output, "report.json")
        write_json(mml_path, mml)
        write_json(report_path, report)
        dxf_path = os.path.join(args.output, "drawing.dxf")
        draw_dxf(mml, dxf_path)
        png_path = os.path.join(args.output, "drawing.png")
        draw_png(mml, png_path)
        stl_path = os.path.join(args.output, "model.stl")
        write_stl(mml, stl_path)
        print(vision_path)
        print(mml_path)
        print(report_path)
        print(dxf_path)
        print(png_path)
        print(stl_path)
        return
    outputs = run_pipeline(args.input, args.output, params=params, api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("OPENAI_MODEL"))
    for key in ["vision", "mml", "report", "dxf"]:
        print(outputs[key])
    print(outputs["png"])
    print(outputs["stl"])


def cmd_library(args):
    """Run library-based pipeline with AI part selection."""
    params = {
        "use_ai": args.ai,
    }
    api_key = os.getenv("OPENAI_API_KEY")

    outputs = run_library_pipeline(
        args.input,
        args.output,
        params=params,
        api_key=api_key,
        model=os.getenv("OPENAI_MODEL"),
        use_ai_selection=args.ai_select
    )

    print(f"Vision: {outputs['vision']}")
    print(f"MML: {outputs['mml']}")
    print(f"Report: {outputs['report']}")
    print(f"Selection: {outputs['selection']}")
    print(f"Drawing: {outputs['dxf']}")
    print(f"Preview: {outputs['png']}")
    print(f"Generated STL files:")
    for stl_path in outputs.get("stl_files", []):
        print(f"  - {stl_path}")
    print(f"\nSelected {len(outputs.get('selected_parts', []))} parts from library")


def build_parser():
    parser = argparse.ArgumentParser(prog="mml")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("-o", "--output", default="out", help="Output directory")
        p.add_argument("--ai", action="store_true", help="Use AI vision (requires OPENAI_API_KEY)")

    vision_p = sub.add_parser("vision", help="Run vision and output vision.json")
    vision_p.add_argument("input")
    add_common(vision_p)
    vision_p.set_defaults(func=cmd_vision)

    interact_p = sub.add_parser("interact", help="Run interact and output mml.json/report.json")
    interact_p.add_argument("input")
    interact_p.add_argument("--chat", default="rule", choices=["rule", "none"])
    interact_p.add_argument("--plate-width-mm", type=float, dest="plate_width_mm")
    interact_p.add_argument("--hole-standard", dest="hole_standard")
    interact_p.add_argument("--hole-diameter-mm", type=float, dest="hole_diameter_mm")
    interact_p.add_argument("--thickness-mm", type=float, dest="thickness_mm")
    interact_p.add_argument("--bend-angle-deg", type=float, dest="bend_angle_deg")
    interact_p.add_argument("--bend-radius-mm", type=float, dest="bend_radius_mm")
    interact_p.add_argument("--unify-holes", action="store_true")
    add_common(interact_p)
    interact_p.set_defaults(func=cmd_interact)

    draw_p = sub.add_parser("draw", help="Draw DXF from mml.json")
    draw_p.add_argument("input")
    draw_p.add_argument("-o", "--output", default="out", help="Output directory")
    draw_p.set_defaults(func=cmd_draw)

    pipeline_p = sub.add_parser("pipeline", help="Run full pipeline")
    pipeline_p.add_argument("input")
    pipeline_p.add_argument("--chat", default="rule", choices=["rule", "none"])
    pipeline_p.add_argument("--plate-width-mm", type=float, dest="plate_width_mm")
    pipeline_p.add_argument("--hole-standard", dest="hole_standard")
    pipeline_p.add_argument("--hole-diameter-mm", type=float, dest="hole_diameter_mm")
    pipeline_p.add_argument("--thickness-mm", type=float, dest="thickness_mm")
    pipeline_p.add_argument("--bend-angle-deg", type=float, dest="bend_angle_deg")
    pipeline_p.add_argument("--bend-radius-mm", type=float, dest="bend_radius_mm")
    pipeline_p.add_argument("--unify-holes", action="store_true")
    add_common(pipeline_p)
    pipeline_p.set_defaults(func=cmd_pipeline)

    # ライブラリベースのパイプライン
    library_p = sub.add_parser("library", help="Run library-based pipeline with part selection")
    library_p.add_argument("input", help="Input image path")
    library_p.add_argument("-o", "--output", default="out", help="Output directory")
    library_p.add_argument("--ai", action="store_true", help="Use AI vision")
    library_p.add_argument("--ai-select", action="store_true", dest="ai_select",
                          help="Use AI for part selection (requires OPENAI_API_KEY)")
    library_p.set_defaults(func=cmd_library)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
