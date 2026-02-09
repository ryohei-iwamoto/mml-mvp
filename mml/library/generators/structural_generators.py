"""
構造部品のメッシュ生成関数。
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


def _rounded_rect_outline(
    width: float,
    height: float,
    corner_radius: float,
    segments_per_corner: int = 8
) -> List[Tuple[float, float]]:
    """角丸矩形の輪郭を生成する。"""
    points = []
    r = min(corner_radius, width / 2, height / 2)

    corners = [
        (width - r, height - r, 0),      # 右上
        (r, height - r, 90),             # 左上
        (r, r, 180),                     # 左下
        (width - r, r, 270),             # 右下
    ]

    for cx, cy, start_angle in corners:
        for i in range(segments_per_corner + 1):
            angle = math.radians(start_angle + 90 * i / segments_per_corner)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))

    return points


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


def generate_bracket(
    width_mm: float = 60.0,
    height_mm: float = 40.0,
    depth_mm: float = 40.0,
    thickness_mm: float = 3.0,
    hole_diameter_mm: float = 5.0,
    corner_radius_mm: float = 3.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    L字ブラケットのメッシュを生成する。
    """
    meshes = []

    # 水平フランジ
    h_outline = _rounded_rect_outline(width_mm, depth_mm, corner_radius_mm)
    h_holes = []

    # 水平フランジに取付穴を追加
    hole_radius = hole_diameter_mm / 2.0
    margin = 10.0
    h_holes.append(_circle_points((margin, depth_mm / 2), hole_radius, 32))
    h_holes.append(_circle_points((width_mm - margin, depth_mm / 2), hole_radius, 32))

    h_mesh = _extrude_profile(h_outline, h_holes, thickness_mm)
    if h_mesh:
        meshes.append(h_mesh)

    # 垂直フランジ
    v_outline = _rounded_rect_outline(width_mm, height_mm, corner_radius_mm)
    v_holes = []
    v_holes.append(_circle_points((margin, height_mm / 2), hole_radius, 32))
    v_holes.append(_circle_points((width_mm - margin, height_mm / 2), hole_radius, 32))

    v_mesh = _extrude_profile(v_outline, v_holes, thickness_mm)
    if v_mesh:
        # 垂直に回転して配置
        v_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            math.radians(90), [1, 0, 0]
        ))
        v_mesh.apply_translation((0, thickness_mm, thickness_mm))
        meshes.append(v_mesh)

    if not meshes:
        # 簡易L字形状にフォールバック
        return trimesh.creation.box(extents=(width_mm, depth_mm, thickness_mm))

    return trimesh.util.concatenate(meshes)


def generate_plate(
    width_mm: float = 100.0,
    height_mm: float = 60.0,
    thickness_mm: float = 3.0,
    corner_radius_mm: float = 3.0,
    hole_pattern: str = "corners",
    hole_diameter_mm: float = 5.0,
    hole_margin_mm: float = 10.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    穴付き矩形プレートのメッシュを生成する。
    """
    outline = _rounded_rect_outline(width_mm, height_mm, corner_radius_mm)
    holes = []

    hole_radius = hole_diameter_mm / 2.0
    margin = hole_margin_mm

    if hole_pattern == "corners":
        # 四隅の穴
        positions = [
            (margin, margin),
            (width_mm - margin, margin),
            (margin, height_mm - margin),
            (width_mm - margin, height_mm - margin),
        ]
        for pos in positions:
            holes.append(_circle_points(pos, hole_radius, 32))

    elif hole_pattern == "edges":
        # 辺沿いの穴
        positions = [
            (margin, height_mm / 2),
            (width_mm - margin, height_mm / 2),
            (width_mm / 2, margin),
            (width_mm / 2, height_mm - margin),
        ]
        for pos in positions:
            holes.append(_circle_points(pos, hole_radius, 32))

    elif hole_pattern == "center":
        # 中央の単一穴
        holes.append(_circle_points((width_mm / 2, height_mm / 2), hole_radius, 32))

    mesh = _extrude_profile(outline, holes, thickness_mm)
    if mesh is None:
        mesh = trimesh.creation.box(extents=(width_mm, height_mm, thickness_mm))
        mesh.apply_translation((width_mm / 2, height_mm / 2, thickness_mm / 2))

    return mesh


def generate_frame(
    outer_width_mm: float = 120.0,
    outer_height_mm: float = 80.0,
    wall_thickness_mm: float = 10.0,
    depth_mm: float = 5.0,
    corner_radius_mm: float = 5.0,
    mounting_holes: bool = True,
    hole_diameter_mm: float = 5.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    矩形フレーム（中空矩形）を生成する。
    """
    outer_outline = _rounded_rect_outline(outer_width_mm, outer_height_mm, corner_radius_mm)

    inner_width = outer_width_mm - 2 * wall_thickness_mm
    inner_height = outer_height_mm - 2 * wall_thickness_mm
    inner_radius = max(1.0, corner_radius_mm - wall_thickness_mm / 2)

    holes = []

    # 内側の抜き
    if inner_width > 0 and inner_height > 0:
        inner_outline = _rounded_rect_outline(inner_width, inner_height, inner_radius)
        # 内側輪郭を中央へオフセット
        inner_outline = [(x + wall_thickness_mm, y + wall_thickness_mm) for x, y in inner_outline]
        holes.append(inner_outline)

    # 取付穴
    if mounting_holes:
        hole_radius = hole_diameter_mm / 2.0
        margin = wall_thickness_mm / 2
        positions = [
            (margin, margin),
            (outer_width_mm - margin, margin),
            (margin, outer_height_mm - margin),
            (outer_width_mm - margin, outer_height_mm - margin),
        ]
        for pos in positions:
            holes.append(_circle_points(pos, hole_radius, 32))

    mesh = _extrude_profile(outer_outline, holes, depth_mm)
    if mesh is None:
        mesh = trimesh.creation.box(extents=(outer_width_mm, outer_height_mm, depth_mm))
        mesh.apply_translation((outer_width_mm / 2, outer_height_mm / 2, depth_mm / 2))

    return mesh
