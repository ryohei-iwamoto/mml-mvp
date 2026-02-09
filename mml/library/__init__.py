"""
部品ライブラリモジュール - 公開API

このモジュールは以下の主要インターフェースを提供する:
1. 部品カタログの読み込みと検索
2. AIによる部品選択
3. 定義からのメッシュ生成
"""

from .catalog import PartsCatalog, PartDefinition, get_catalog
from .selector import PartSelector, select_parts_for_intent
from .generator import generate_part_mesh, PartGenerationResult
from .validators import validate_parameters, fill_defaults

__all__ = [
    # カタログ
    "PartsCatalog",
    "PartDefinition",
    "get_catalog",
    # 選択
    "PartSelector",
    "select_parts_for_intent",
    # 生成
    "generate_part_mesh",
    "PartGenerationResult",
    # 検証
    "validate_parameters",
    "fill_defaults",
]
