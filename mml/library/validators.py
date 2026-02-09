"""
部品定義のパラメータ検証ユーティリティ。
"""

from typing import Dict, Any, List, Optional
from .catalog import PartDefinition


def fill_defaults(
    definition: PartDefinition, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    欠落パラメータに既定値を補完する。

    引数:
        definition: パラメータ仕様を含む部品定義
        params: ユーザー指定のパラメータ

    戻り値:
        既定値を補完したパラメータ辞書
    """
    result = {}
    for param_name, param_def in definition.parameters.items():
        if param_name in params and params[param_name] is not None:
            result[param_name] = params[param_name]
        else:
            result[param_name] = param_def.get("default")
    return result


def validate_parameters(
    definition: PartDefinition, params: Dict[str, Any]
) -> List[str]:
    """
    パラメータを部品定義の制約に対して検証する。

    引数:
        definition: パラメータ仕様を含む部品定義
        params: 検証対象のパラメータ

    戻り値:
        警告メッセージのリスト（問題なければ空）
    """
    warnings = []

    for param_name, param_def in definition.parameters.items():
        value = params.get(param_name)
        if value is None:
            continue

        param_type = param_def.get("type", "float")

        # 型の検証
        if param_type == "int":
            if not isinstance(value, (int, float)):
                warnings.append(f"{param_name}: expected int, got {type(value).__name__}")
            elif isinstance(value, float) and value != int(value):
                warnings.append(f"{param_name}: expected int, got float {value}")

        # 範囲の検証
        min_val = param_def.get("min")
        max_val = param_def.get("max")
        if min_val is not None and value < min_val:
            warnings.append(f"{param_name}: value {value} below minimum {min_val}")
        if max_val is not None and value > max_val:
            warnings.append(f"{param_name}: value {value} above maximum {max_val}")

        # 列挙値の検証
        enum_values = param_def.get("enum_values")
        if enum_values is not None and value not in enum_values:
            warnings.append(f"{param_name}: value {value} not in allowed values {enum_values}")

    # 式による制約
    for constraint in definition.constraints:
        if constraint.get("kind") == "expression":
            expr = constraint.get("expression", "")
            try:
                # params をローカルとして簡易式評価
                if not eval(expr, {"__builtins__": {}}, params):
                    msg = constraint.get("message", f"Constraint failed: {expr}")
                    warnings.append(msg)
            except Exception:
                pass  # 不正な式は無視

    return warnings


def coerce_parameters(
    definition: PartDefinition, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    パラメータ値を期待される型に変換する。

    引数:
        definition: パラメータ仕様を含む部品定義
        params: 変換対象のパラメータ

    戻り値:
        型変換後のパラメータ
    """
    result = {}
    for param_name, value in params.items():
        if param_name not in definition.parameters:
            result[param_name] = value
            continue

        param_def = definition.parameters[param_name]
        param_type = param_def.get("type", "float")

        if value is None:
            result[param_name] = None
        elif param_type == "int":
            result[param_name] = int(value)
        elif param_type == "float":
            result[param_name] = float(value)
        elif param_type == "bool":
            result[param_name] = bool(value)
        elif param_type == "string":
            result[param_name] = str(value)
        else:
            result[param_name] = value

    return result


def apply_scale(
    definition: PartDefinition, params: Dict[str, Any], scale: float
) -> Dict[str, Any]:
    """
    寸法パラメータにスケール係数を適用する。

    引数:
        definition: パラメータ仕様を含む部品定義
        params: スケール適用対象のパラメータ
        scale: スケール係数

    戻り値:
        寸法をスケールしたパラメータ
    """
    if scale == 1.0:
        return params

    scaled = params.copy()
    for param_name, param_def in definition.parameters.items():
        unit = param_def.get("unit", "")
        if unit in ("mm", "m", "cm", "inch"):
            if param_name in scaled and scaled[param_name] is not None:
                scaled[param_name] = scaled[param_name] * scale

    return scaled
