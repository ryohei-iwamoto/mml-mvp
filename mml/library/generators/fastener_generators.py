"""
締結部品のメッシュ生成関数。
"""

import math
from typing import List, Tuple, Optional

import trimesh
from shapely.geometry import Polygon


def _circle_points(center: Tuple[float, float], radius: float, segments: int = 48) -> List[Tuple[float, float]]:
    """円周上の点を生成する。"""
    cx, cy = center
    pts = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts


def _hexagon_points(center: Tuple[float, float], flat_to_flat: float) -> List[Tuple[float, float]]:
    """六角形の点列を生成する（角-角方向）。"""
    cx, cy = center
    # 六角ナットでは対辺距離が二面幅
    # 角までの半径 = 対辺距離 / sqrt(3)
    radius = flat_to_flat / math.sqrt(3)
    pts = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3  # 30度から開始
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        pts.append((x, y))
    return pts


def _extrude_profile(
    outline: List[Tuple[float, float]],
    holes: List[List[Tuple[float, float]]],
    height: float,
    z_offset: float = 0.0
) -> Optional[trimesh.Trimesh]:
    """穴付き2Dプロファイルを3Dに押し出す。"""
    poly = Polygon(outline, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None

    try:
        mesh = trimesh.creation.extrude_polygon(
            poly, height=height,
            triangulate_kwargs={"engine": "earcut"}
        )
        if z_offset:
            mesh.apply_translation((0.0, 0.0, z_offset))
        return mesh
    except Exception:
        return None


# メトリックボルト/ナットの標準寸法（概略）
METRIC_FASTENERS = {
    "M3": {"pitch": 0.5, "head_dia": 5.5, "head_height": 2.0, "nut_flat": 5.5, "nut_height": 2.4},
    "M4": {"pitch": 0.7, "head_dia": 7.0, "head_height": 2.8, "nut_flat": 7.0, "nut_height": 3.2},
    "M5": {"pitch": 0.8, "head_dia": 8.5, "head_height": 3.5, "nut_flat": 8.0, "nut_height": 4.0},
    "M6": {"pitch": 1.0, "head_dia": 10.0, "head_height": 4.0, "nut_flat": 10.0, "nut_height": 5.0},
    "M8": {"pitch": 1.25, "head_dia": 13.0, "head_height": 5.3, "nut_flat": 13.0, "nut_height": 6.5},
    "M10": {"pitch": 1.5, "head_dia": 16.0, "head_height": 6.4, "nut_flat": 17.0, "nut_height": 8.0},
    "M12": {"pitch": 1.75, "head_dia": 18.0, "head_height": 7.5, "nut_flat": 19.0, "nut_height": 10.0},
}


def generate_bolt(
    size: str = "M5",
    length_mm: float = 20.0,
    head_type: str = "hex",
    thread_length_mm: Optional[float] = None,
    **kwargs
) -> trimesh.Trimesh:
    """
    ボルトのメッシュを生成する。

    引数:
        size: 呼び径（M3, M4, M5, M6, M8, M10, M12）
        length_mm: 全長（頭部を除く）
        head_type: "hex", "socket", "pan"
        thread_length_mm: ねじ長さ（Noneで全ねじ）
    """
    # サイズに対応する寸法を取得
    dims = METRIC_FASTENERS.get(size.upper(), METRIC_FASTENERS["M5"])
    nominal_dia = float(size.upper().replace("M", ""))

    meshes = []

    # 頭部
    if head_type == "hex":
        head_outline = _hexagon_points((0, 0), dims["head_dia"])
        head_mesh = _extrude_profile(head_outline, [], dims["head_height"])
        if head_mesh is None:
            head_mesh = trimesh.creation.cylinder(
                radius=dims["head_dia"] / 2.0,
                height=dims["head_height"],
                sections=6
            )
    elif head_type == "socket":
        # 六角穴付きボルト頭部
        head_dia = nominal_dia * 1.5
        head_height = nominal_dia
        head_mesh = trimesh.creation.cylinder(
            radius=head_dia / 2.0,
            height=head_height,
            sections=64
        )
        # ソケット穴を追加
        socket_dia = nominal_dia * 0.9
        socket_depth = head_height * 0.6
        socket = trimesh.creation.cylinder(
            radius=socket_dia / 2.0,
            height=socket_depth,
            sections=6  # 六角穴
        )
        socket.apply_translation((0, 0, head_height - socket_depth / 2.0))
        try:
            head_mesh = head_mesh.difference(socket, engine="blender")
        except Exception:
            pass
    else:  # 皿頭
        head_dia = nominal_dia * 2.0
        head_height = nominal_dia * 0.6
        head_mesh = trimesh.creation.cylinder(
            radius=head_dia / 2.0,
            height=head_height,
            sections=64
        )

    head_mesh.apply_translation((0, 0, head_mesh.bounds[1][2] - head_mesh.bounds[0][2]))
    meshes.append(head_mesh)

    # 軸部/ねじ部
    shank = trimesh.creation.cylinder(
        radius=nominal_dia / 2.0,
        height=length_mm,
        sections=32
    )
    head_top = head_mesh.bounds[1][2]
    shank.apply_translation((0, 0, head_top + length_mm / 2.0))
    meshes.append(shank)

    return trimesh.util.concatenate(meshes)


def generate_nut(
    size: str = "M5",
    nut_type: str = "hex",
    nyloc: bool = False,
    **kwargs
) -> trimesh.Trimesh:
    """
    ナットのメッシュを生成する。

    引数:
        size: 呼び径（M3, M4, M5, M6, M8, M10, M12）
        nut_type: "hex", "square", "flange"
        nyloc: ナイロンインサートを追加
    """
    dims = METRIC_FASTENERS.get(size.upper(), METRIC_FASTENERS["M5"])
    nominal_dia = float(size.upper().replace("M", ""))

    nut_height = dims["nut_height"]
    if nyloc:
        nut_height *= 1.3

    if nut_type == "hex":
        outline = _hexagon_points((0, 0), dims["nut_flat"])
    elif nut_type == "square":
        s = dims["nut_flat"]
        outline = [(-s/2, -s/2), (s/2, -s/2), (s/2, s/2), (-s/2, s/2)]
    else:  # フランジ
        outline = _hexagon_points((0, 0), dims["nut_flat"])

    holes = [_circle_points((0, 0), nominal_dia / 2.0, 32)]

    mesh = _extrude_profile(outline, holes, nut_height)

    if mesh is None:
        mesh = trimesh.creation.cylinder(
            radius=dims["nut_flat"] / 2.0,
            height=nut_height,
            sections=6
        )

    # 指定時にフランジを追加
    if nut_type == "flange":
        flange_dia = dims["nut_flat"] * 1.4
        flange_height = nut_height * 0.2
        flange_outline = _circle_points((0, 0), flange_dia / 2.0, 64)
        flange_holes = [_circle_points((0, 0), nominal_dia / 2.0, 32)]
        flange = _extrude_profile(flange_outline, flange_holes, flange_height)
        if flange:
            mesh = trimesh.util.concatenate([mesh, flange])

    return mesh


def generate_spacer(
    outer_diameter_mm: float = 10.0,
    inner_diameter_mm: float = 5.0,
    length_mm: float = 5.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    スペーサ/スタンドオフのメッシュを生成する。
    """
    outline = _circle_points((0, 0), outer_diameter_mm / 2.0, 64)
    holes = []

    if inner_diameter_mm > 0:
        holes.append(_circle_points((0, 0), inner_diameter_mm / 2.0, 32))

    mesh = _extrude_profile(outline, holes, length_mm)

    if mesh is None:
        outer = trimesh.creation.cylinder(
            radius=outer_diameter_mm / 2.0,
            height=length_mm,
            sections=64
        )
        if inner_diameter_mm > 0:
            inner = trimesh.creation.cylinder(
                radius=inner_diameter_mm / 2.0,
                height=length_mm * 1.1,
                sections=32
            )
            try:
                mesh = outer.difference(inner, engine="blender")
            except Exception:
                mesh = outer
        else:
            mesh = outer

    return mesh


def generate_washer(
    size: str = "M5",
    washer_type: str = "flat",
    **kwargs
) -> trimesh.Trimesh:
    """
    ワッシャのメッシュを生成する。

    引数:
        size: 呼び径（穴径の基準）
        washer_type: "flat", "spring", "lock"
    """
    nominal_dia = float(size.upper().replace("M", ""))

    # ワッシャ標準寸法（概略）
    inner_dia = nominal_dia + 0.3  # クリアランス
    outer_dia = nominal_dia * 2.2
    thickness = nominal_dia * 0.2

    if washer_type == "spring":
        # スプリング/ロックワッシャ
        thickness = nominal_dia * 0.25
        outer_dia = nominal_dia * 2.0

    outline = _circle_points((0, 0), outer_dia / 2.0, 64)
    holes = [_circle_points((0, 0), inner_dia / 2.0, 32)]

    mesh = _extrude_profile(outline, holes, thickness)

    if mesh is None:
        mesh = trimesh.creation.annulus(
            r_min=inner_dia / 2.0,
            r_max=outer_dia / 2.0,
            height=thickness
        )

    return mesh
