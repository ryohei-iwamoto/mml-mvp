"""
部品メッシュ生成モジュール。

各部品種別の3Dメッシュ生成関数を含む。
各関数は部品定義に対応するキーワード引数を受け取り、
trimesh.Trimesh を返す。
"""

from .gear_generators import (
    generate_spur_gear,
    generate_helical_gear,
    generate_bevel_gear,
    generate_rack,
)
from .structural_generators import (
    generate_bracket,
    generate_plate,
    generate_frame,
)
from .drive_generators import (
    generate_motor,
    generate_shaft,
    generate_bearing,
    generate_coupling,
)
from .fastener_generators import (
    generate_bolt,
    generate_nut,
    generate_spacer,
    generate_washer,
)

__all__ = [
    # 歯車
    "generate_spur_gear",
    "generate_helical_gear",
    "generate_bevel_gear",
    "generate_rack",
    # 構造部品
    "generate_bracket",
    "generate_plate",
    "generate_frame",
    # 駆動部品
    "generate_motor",
    "generate_shaft",
    "generate_bearing",
    "generate_coupling",
    # 締結部品
    "generate_bolt",
    "generate_nut",
    "generate_spacer",
    "generate_washer",
]
