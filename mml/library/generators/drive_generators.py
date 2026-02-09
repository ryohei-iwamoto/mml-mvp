"""
駆動部品のメッシュ生成関数。
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


def generate_motor(
    body_diameter_mm: float = 42.0,
    body_length_mm: float = 40.0,
    shaft_diameter_mm: float = 5.0,
    shaft_length_mm: float = 20.0,
    flange_diameter_mm: float = 50.0,
    flange_thickness_mm: float = 3.0,
    mounting_hole_diameter_mm: float = 3.0,
    mounting_hole_pcd_mm: float = 31.0,
    mounting_holes_count: int = 4,
    **kwargs
) -> trimesh.Trimesh:
    """
    モータのメッシュを生成する（NEMA風）。
    """
    meshes = []

    # 本体円筒
    body = trimesh.creation.cylinder(
        radius=body_diameter_mm / 2.0,
        height=body_length_mm,
        sections=64
    )
    body.apply_translation((0, 0, body_length_mm / 2.0))
    meshes.append(body)

    # 前面フランジ
    flange_outline = _circle_points((0, 0), flange_diameter_mm / 2.0, 64)
    flange_holes = []

    # PCD上の取付穴
    if mounting_holes_count > 0 and mounting_hole_pcd_mm > 0:
        hole_radius = mounting_hole_diameter_mm / 2.0
        pcd_radius = mounting_hole_pcd_mm / 2.0
        for i in range(mounting_holes_count):
            angle = 2 * math.pi * i / mounting_holes_count + math.pi / 4  # 45度オフセット
            hx = pcd_radius * math.cos(angle)
            hy = pcd_radius * math.sin(angle)
            flange_holes.append(_circle_points((hx, hy), hole_radius, 32))

    flange_mesh = _extrude_profile(flange_outline, flange_holes, flange_thickness_mm)
    if flange_mesh:
        flange_mesh.apply_translation((0, 0, body_length_mm))
        meshes.append(flange_mesh)

    # 出力軸
    if shaft_diameter_mm > 0 and shaft_length_mm > 0:
        shaft = trimesh.creation.cylinder(
            radius=shaft_diameter_mm / 2.0,
            height=shaft_length_mm,
            sections=32
        )
        shaft.apply_translation((0, 0, body_length_mm + flange_thickness_mm + shaft_length_mm / 2.0))
        meshes.append(shaft)

    return trimesh.util.concatenate(meshes)


def generate_shaft(
    diameter_mm: float = 8.0,
    length_mm: float = 100.0,
    keyway: bool = False,
    keyway_width_mm: float = 3.0,
    keyway_depth_mm: float = 1.5,
    keyway_length_mm: float = 20.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    軸のメッシュを生成する。
    """
    shaft = trimesh.creation.cylinder(
        radius=diameter_mm / 2.0,
        height=length_mm,
        sections=64
    )
    shaft.apply_translation((0, 0, length_mm / 2.0))

    if keyway:
        # キー溝の切り欠きを作成
        keyway_mesh = trimesh.creation.box(
            extents=(keyway_width_mm, diameter_mm, keyway_length_mm)
        )
        # 軸上端に配置
        keyway_mesh.apply_translation((
            0,
            diameter_mm / 2.0 - keyway_depth_mm / 2.0,
            length_mm - keyway_length_mm / 2.0
        ))
        try:
            shaft = shaft.difference(keyway_mesh, engine="blender")
        except Exception:
            pass  # ブーリアン失敗時はキー溝なしで返す

    return shaft


def generate_bearing(
    outer_diameter_mm: float = 22.0,
    inner_diameter_mm: float = 8.0,
    width_mm: float = 7.0,
    chamfer_mm: float = 0.5,
    **kwargs
) -> trimesh.Trimesh:
    """
    玉軸受のメッシュを生成する（簡略）。
    """
    outer_outline = _circle_points((0, 0), outer_diameter_mm / 2.0, 64)
    holes = []

    # 内径ボア
    if inner_diameter_mm > 0:
        holes.append(_circle_points((0, 0), inner_diameter_mm / 2.0, 64))

    mesh = _extrude_profile(outer_outline, holes, width_mm)

    if mesh is None:
        # フォールバック
        outer = trimesh.creation.cylinder(
            radius=outer_diameter_mm / 2.0,
            height=width_mm,
            sections=64
        )
        inner = trimesh.creation.cylinder(
            radius=inner_diameter_mm / 2.0,
            height=width_mm * 1.1,
            sections=64
        )
        try:
            mesh = outer.difference(inner, engine="blender")
        except Exception:
            mesh = outer

    return mesh


def generate_coupling(
    outer_diameter_mm: float = 25.0,
    bore_diameter_mm: float = 8.0,
    length_mm: float = 30.0,
    coupling_type: str = "rigid",
    jaw_count: int = 3,
    **kwargs
) -> trimesh.Trimesh:
    """
    軸継手のメッシュを生成する。
    """
    if coupling_type == "jaw" or coupling_type == "spider":
        # ジョーカップリング（スパイダー）
        return _generate_jaw_coupling(
            outer_diameter_mm, bore_diameter_mm, length_mm, jaw_count
        )
    else:
        # 簡易リジッドカップリング
        return _generate_rigid_coupling(
            outer_diameter_mm, bore_diameter_mm, length_mm
        )


def _generate_rigid_coupling(
    outer_diameter_mm: float,
    bore_diameter_mm: float,
    length_mm: float
) -> trimesh.Trimesh:
    """簡易リジッドカップリングを生成する。"""
    outer_outline = _circle_points((0, 0), outer_diameter_mm / 2.0, 64)
    holes = []

    if bore_diameter_mm > 0:
        holes.append(_circle_points((0, 0), bore_diameter_mm / 2.0, 64))

    mesh = _extrude_profile(outer_outline, holes, length_mm)

    if mesh is None:
        mesh = trimesh.creation.cylinder(
            radius=outer_diameter_mm / 2.0,
            height=length_mm,
            sections=64
        )

    return mesh


def _generate_jaw_coupling(
    outer_diameter_mm: float,
    bore_diameter_mm: float,
    length_mm: float,
    jaw_count: int
) -> trimesh.Trimesh:
    """ジョーカップリング片側を生成する。"""
    meshes = []

    # 基本円筒
    base_height = length_mm * 0.3
    base_outline = _circle_points((0, 0), outer_diameter_mm / 2.0, 64)
    base_holes = []
    if bore_diameter_mm > 0:
        base_holes.append(_circle_points((0, 0), bore_diameter_mm / 2.0, 64))

    base_mesh = _extrude_profile(base_outline, base_holes, base_height)
    if base_mesh:
        meshes.append(base_mesh)

    # ジョー歯
    jaw_height = length_mm * 0.7
    jaw_radius = outer_diameter_mm / 2.0 * 0.85
    jaw_width = 2 * math.pi * jaw_radius / (jaw_count * 2) * 0.8

    for i in range(jaw_count):
        angle = 2 * math.pi * i / jaw_count
        jaw = trimesh.creation.box(
            extents=(jaw_width, outer_diameter_mm * 0.3, jaw_height)
        )
        jaw.apply_translation((
            jaw_radius * math.cos(angle),
            jaw_radius * math.sin(angle),
            base_height + jaw_height / 2.0
        ))
        jaw.apply_transform(trimesh.transformations.rotation_matrix(
            angle, [0, 0, 1]
        ))
        meshes.append(jaw)

    if not meshes:
        return _generate_rigid_coupling(outer_diameter_mm, bore_diameter_mm, length_mm)

    return trimesh.util.concatenate(meshes)
