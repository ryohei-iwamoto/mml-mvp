"""
MML処理のパイプラインモジュール。

従来パイプラインとライブラリベースのパイプラインの両方をサポートする。
"""

import os
from typing import Dict, Any, Optional

from .ai_vision import run_ai_vision
from .draw import draw_dxf, draw_png
from .stl import write_stl
from .emit import emit_mml
from .utils import ensure_dir, write_json
from .vision import normalize_vision, run_vision


def run_pipeline(image_path, out_dir, params=None, api_key=None, model=None):
    params = params or {}
    ensure_dir(out_dir)

    use_ai = bool(params.get("use_ai"))
    if use_ai:
        if not api_key:
            raise ValueError("OPENAI_API_KEY が未設定です。 .env を作成してください。")
        vision = run_ai_vision(image_path, api_key=api_key, model=model)
    else:
        vision = run_vision(image_path)
    vision = normalize_vision(vision)
    vision_path = os.path.join(out_dir, "vision.json")
    write_json(vision_path, vision)

    mml, report = emit_mml(vision, params, os.path.basename(image_path))
    mml_path = os.path.join(out_dir, "mml.json")
    report_path = os.path.join(out_dir, "report.json")
    write_json(mml_path, mml)
    write_json(report_path, report)

    dxf_path = os.path.join(out_dir, "drawing.dxf")
    draw_dxf(mml, dxf_path)
    png_path = os.path.join(out_dir, "drawing.png")
    draw_png(mml, png_path)
    stl_path = os.path.join(out_dir, "model.stl")
    write_stl(mml, stl_path)

    return {
        "vision": vision_path,
        "mml": mml_path,
        "report": report_path,
        "dxf": dxf_path,
        "png": png_path,
        "stl": stl_path,
    }


def run_library_pipeline(
    image_path: str,
    out_dir: str,
    params: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_ai_selection: bool = True
) -> Dict[str, Any]:
    """
    部品ライブラリを用いた生成パイプライン。

    引数:
        image_path: 入力画像のパス
        out_dir: 出力ディレクトリ
        params: ユーザーパラメータ
        api_key: OpenAI API キー
        model: OpenAI モデル名
        use_ai_selection: 部品選択にAIを使うかどうか

    戻り値:
        出力ファイルパスと選択結果の辞書
    """
    from .intent import infer_parts_from_intent
    from .library import generate_part_mesh

    params = params or {}
    ensure_dir(out_dir)

    # 手順1: ビジョン解析
    use_ai = bool(params.get("use_ai"))
    if use_ai and api_key:
        vision = run_ai_vision(image_path, api_key=api_key, model=model)
    else:
        vision = run_vision(image_path)
    vision = normalize_vision(vision)
    vision_path = os.path.join(out_dir, "vision.json")
    write_json(vision_path, vision)

    # 手順2: 意図付きMML生成
    mml, report = emit_mml(
        vision, params, os.path.basename(image_path),
        include_intent=True
    )

    # 手順3: ライブラリから部品選択
    intent = mml.get("intent", {})
    selection_api_key = api_key if use_ai_selection else None
    selected_parts = infer_parts_from_intent(intent, vision, selection_api_key)

    # 選択結果を保存
    selection_path = os.path.join(out_dir, "parts_selection.json")
    write_json(selection_path, {
        "selected_parts": selected_parts,
        "catalog_version": "1.0"
    })

    # 手順4: 選択部品のメッシュ生成
    stl_paths = []
    generation_errors = []

    for part_info in selected_parts:
        part_id = part_info["part_id"]
        part_params = part_info.get("parameters", {})
        quantity = part_info.get("quantity", 1)

        try:
            result = generate_part_mesh(part_id, part_params)

            for q in range(quantity):
                suffix = f"_{q+1}" if quantity > 1 else ""
                stl_name = f"{part_id}{suffix}.stl"
                stl_path = os.path.join(out_dir, stl_name)
                result.mesh.export(stl_path)
                stl_paths.append(stl_path)

        except Exception as e:
            generation_errors.append(f"{part_id}: {str(e)}")

    if generation_errors:
        report["generation_errors"] = generation_errors

    # 手順5: MMLとレポートを保存
    mml_path = os.path.join(out_dir, "mml.json")
    report_path = os.path.join(out_dir, "report.json")
    write_json(mml_path, mml)
    write_json(report_path, report)

    # 手順6: 図面生成
    dxf_path = os.path.join(out_dir, "drawing.dxf")
    draw_dxf(mml, dxf_path)
    png_path = os.path.join(out_dir, "drawing.png")
    draw_png(mml, png_path)

    return {
        "vision": vision_path,
        "mml": mml_path,
        "report": report_path,
        "selection": selection_path,
        "dxf": dxf_path,
        "png": png_path,
        "stl_files": stl_paths,
        "selected_parts": selected_parts
    }
