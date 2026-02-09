"""
部品定義からメッシュを生成する。

定義に基づき適切な生成関数へディスパッチする。
"""

import importlib
from typing import Dict, Any, Optional
from dataclasses import dataclass

import trimesh

from .catalog import PartDefinition, get_catalog
from .validators import validate_parameters, fill_defaults, coerce_parameters, apply_scale


@dataclass
class PartGenerationResult:
    """部品生成結果。"""

    mesh: trimesh.Trimesh
    part_id: str
    parameters_used: Dict[str, Any]
    warnings: list


def generate_part_mesh(
    part_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    scale: float = 1.0
) -> PartGenerationResult:
    """
    ライブラリから部品メッシュを生成する。

    引数:
        part_id: カタログ内の部品ID
        parameters: パラメータ値（欠落は既定値を使用）
        scale: 全体スケール係数

    戻り値:
        生成メッシュを含むPartGenerationResult

    例外:
        ValueError: カタログに部品IDが存在しない場合
        RuntimeError: 生成に失敗した場合
    """
    catalog = get_catalog()
    definition = catalog.get(part_id)

    if definition is None:
        raise ValueError(f"Unknown part ID: {part_id}")

    # 既定値補完・型変換・検証
    params = fill_defaults(definition, parameters or {})
    params = coerce_parameters(definition, params)
    warnings = validate_parameters(definition, params)

    # 寸法パラメータにスケールを適用
    params = apply_scale(definition, params, scale)

    # 生成関数をインポートして実行
    try:
        module = importlib.import_module(definition.generator["module"])
        func = getattr(module, definition.generator["function"])
    except (ModuleNotFoundError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to load generator for {part_id}: "
            f"{definition.generator['module']}.{definition.generator['function']}"
        ) from e

    try:
        mesh = func(**params)
    except Exception as e:
        raise RuntimeError(f"Failed to generate mesh for {part_id}: {e}") from e

    return PartGenerationResult(
        mesh=mesh,
        part_id=part_id,
        parameters_used=params,
        warnings=warnings,
    )


def generate_assembly(
    parts: list,
    scale: float = 1.0
) -> Dict[str, Any]:
    """
    複数部品のメッシュを生成する。

    引数:
        parts: 'part_id', 'parameters', 'quantity'(任意)を含む辞書のリスト
        scale: 全体スケール係数

    戻り値:
        'meshes' と 'warnings' を含む辞書
    """
    results = {
        "meshes": [],
        "warnings": [],
    }

    for part_info in parts:
        part_id = part_info.get("part_id")
        params = part_info.get("parameters", {})
        quantity = part_info.get("quantity", 1)

        try:
            result = generate_part_mesh(part_id, params, scale)

            for i in range(quantity):
                results["meshes"].append({
                    "part_id": part_id,
                    "instance": i + 1,
                    "mesh": result.mesh.copy(),
                    "parameters": result.parameters_used,
                })

            if result.warnings:
                results["warnings"].extend(
                    f"{part_id}: {w}" for w in result.warnings
                )

        except Exception as e:
            results["warnings"].append(f"{part_id}: Generation failed - {e}")

    return results


def list_available_generators() -> Dict[str, str]:
    """
    利用可能な生成関数を一覧する。

    戻り値:
        part_id と生成関数パスの対応辞書
    """
    catalog = get_catalog()
    return {
        part.id: f"{part.generator['module']}.{part.generator['function']}"
        for part in catalog.all_parts()
    }
