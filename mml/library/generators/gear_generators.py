"""
歯車メッシュ生成関数。

これらの関数はライブラリの生成ディスパッチャから呼ばれる。
各関数は部品定義に対応するキーワード引数を受け取る。
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


def _involute_point(base_radius: float, angle: float) -> Tuple[float, float]:
    """インボリュート曲線上の点を計算する。"""
    x = base_radius * (math.cos(angle) + angle * math.sin(angle))
    y = base_radius * (math.sin(angle) - angle * math.cos(angle))
    return (x, y)


def _gear_tooth_profile(
    module: float,
    teeth_count: int,
    pressure_angle_deg: float = 20.0,
    points_per_tooth: int = 8
) -> List[Tuple[float, float]]:
    """
    歯形プロファイル点を生成する。

    メッシュ生成のため簡易インボリュート近似を用いる。
    """
    pitch_diameter = module * teeth_count
    pitch_radius = pitch_diameter / 2.0

    addendum = module
    dedendum = 1.25 * module

    outer_radius = pitch_radius + addendum
    root_radius = max(pitch_radius - dedendum, module)
    base_radius = pitch_radius * math.cos(math.radians(pressure_angle_deg))

    points = []
    tooth_angle = 2 * math.pi / teeth_count

    for i in range(teeth_count):
        base_angle = i * tooth_angle

        # 主要点による簡易歯形プロファイル
        # 歯元 → 歯面 → 歯先 → 歯面 → 歯元
        tooth_half = tooth_angle * 0.25

        angles_and_radii = [
            (-tooth_half * 1.4, root_radius),      # 歯元開始
            (-tooth_half * 0.8, pitch_radius * 0.95),  # 下側歯面
            (-tooth_half * 0.3, outer_radius * 0.98),  # 上側歯面
            (0, outer_radius),                      # 歯先中心
            (tooth_half * 0.3, outer_radius * 0.98),   # 上側歯面
            (tooth_half * 0.8, pitch_radius * 0.95),   # 下側歯面
            (tooth_half * 1.4, root_radius),       # 歯元終端
        ]

        for offset, radius in angles_and_radii:
            angle = base_angle + offset
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
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


def generate_spur_gear(
    module: float = 1.0,
    teeth_count: int = 24,
    pressure_angle_deg: float = 20.0,
    face_width_mm: float = 10.0,
    bore_diameter_mm: float = 8.0,
    hub_diameter_mm: float = 0,
    hub_length_mm: float = 0,
    **kwargs
) -> trimesh.Trimesh:
    """
    平歯車のメッシュを生成する。

    引数:
        module: モジュール（歯の大きさ, mm）
        teeth_count: 歯数
        pressure_angle_deg: 圧力角（度）
        face_width_mm: 歯幅（厚み）
        bore_diameter_mm: 中心ボア径
        hub_diameter_mm: ハブ径（0=ハブなし）
        hub_length_mm: ハブ長さ

    戻り値:
        歯車の trimesh.Trimesh
    """
    # 歯形プロファイルを生成
    profile = _gear_tooth_profile(module, teeth_count, pressure_angle_deg)

    outer_radius = (module * teeth_count) / 2.0 + module

    # ボア穴を作成
    holes = []
    if bore_diameter_mm > 0:
        bore_radius = bore_diameter_mm / 2.0
        holes.append(_circle_points((0, 0), bore_radius, segments=64))

    # 歯車本体を押し出し
    main_mesh = _extrude_profile(profile, holes, face_width_mm)

    if main_mesh is None:
        # 円柱にフォールバック
        main_mesh = trimesh.creation.cylinder(
            radius=outer_radius,
            height=face_width_mm,
            sections=96
        )
        if bore_diameter_mm > 0:
            bore_mesh = trimesh.creation.cylinder(
                radius=bore_diameter_mm / 2.0,
                height=face_width_mm * 1.1,
                sections=64
            )
            main_mesh = main_mesh.difference(bore_mesh, engine="blender")

    meshes = [main_mesh]

    # 指定時はハブを追加
    if hub_diameter_mm > 0 and hub_length_mm > 0:
        hub_radius = hub_diameter_mm / 2.0
        hub_holes = []
        if bore_diameter_mm > 0:
            hub_holes.append(_circle_points(
                (0, 0), bore_diameter_mm / 2.0, segments=64
            ))

        hub_outline = _circle_points((0, 0), hub_radius, segments=64)
        hub_mesh = _extrude_profile(
            hub_outline, hub_holes, hub_length_mm,
            z_offset=face_width_mm
        )
        if hub_mesh:
            meshes.append(hub_mesh)

    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def generate_helical_gear(
    module: float = 1.0,
    teeth_count: int = 24,
    pressure_angle_deg: float = 20.0,
    helix_angle_deg: float = 15.0,
    face_width_mm: float = 15.0,
    bore_diameter_mm: float = 8.0,
    hand: str = "right",
    **kwargs
) -> trimesh.Trimesh:
    """
    はすば歯車のメッシュを生成する。

    簡略化のため平歯車で近似する。
    完全なはすば歯車にはねじり押し出しが必要。
    """
    # MVPでは平歯車で代用
    # 完全実装ではZ軸方向にねじり押し出し
    return generate_spur_gear(
        module=module,
        teeth_count=teeth_count,
        pressure_angle_deg=pressure_angle_deg,
        face_width_mm=face_width_mm,
        bore_diameter_mm=bore_diameter_mm,
        **kwargs
    )


def generate_bevel_gear(
    module: float = 1.5,
    teeth_count: int = 20,
    cone_angle_deg: float = 45.0,
    face_width_mm: float = 12.0,
    bore_diameter_mm: float = 10.0,
    shaft_angle_deg: float = 90.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    かさ歯車のメッシュを生成する。

    メッシュ生成のため簡易的な円錐近似を用いる。
    """
    pitch_diameter = module * teeth_count
    outer_radius = pitch_diameter / 2.0 + module

    # テーパ円柱の近似を作成
    cone_half_angle = math.radians(cone_angle_deg)

    # 台形円錐（フラスタム）として生成
    top_radius = outer_radius - face_width_mm * math.tan(cone_half_angle)
    top_radius = max(top_radius, bore_diameter_mm / 2.0 + 2.0)

    # 円柱セクションでフラスタムを作成
    sections = 8
    meshes = []
    section_height = face_width_mm / sections

    for i in range(sections):
        t = i / sections
        r1 = outer_radius - (outer_radius - top_radius) * t
        r2 = outer_radius - (outer_radius - top_radius) * (t + 1/sections)
        r_avg = (r1 + r2) / 2.0

        section_mesh = trimesh.creation.cylinder(
            radius=r_avg,
            height=section_height,
            sections=64
        )
        section_mesh.apply_translation((0, 0, section_height * (i + 0.5)))
        meshes.append(section_mesh)

    mesh = trimesh.util.concatenate(meshes)

    # ボア穴を追加
    if bore_diameter_mm > 0:
        bore_mesh = trimesh.creation.cylinder(
            radius=bore_diameter_mm / 2.0,
            height=face_width_mm * 1.2,
            sections=64
        )
        bore_mesh.apply_translation((0, 0, face_width_mm / 2.0))
        try:
            mesh = mesh.difference(bore_mesh, engine="blender")
        except Exception:
            pass  # ブーリアン失敗時はボア無しで返す

    return mesh


def generate_rack(
    module: float = 1.5,
    teeth_count: int = 20,
    pressure_angle_deg: float = 20.0,
    face_width_mm: float = 15.0,
    height_mm: float = 20.0,
    mounting_holes: bool = True,
    hole_diameter_mm: float = 5.0,
    **kwargs
) -> trimesh.Trimesh:
    """
    ラックのメッシュを生成する。
    """
    pitch = math.pi * module
    length = pitch * teeth_count

    # 歯寸法を計算
    addendum = module
    dedendum = 1.25 * module
    tooth_height = addendum + dedendum

    # 基本プロファイル（ラック側面）
    base_height = height_mm - tooth_height

    # 歯形プロファイル点を生成（2D断面）
    profile_points = []

    # 左下
    profile_points.append((0, 0))
    # 右下
    profile_points.append((length, 0))
    # 右上（歯の基部）
    profile_points.append((length, base_height))

    # 歯を追加
    pressure_angle_rad = math.radians(pressure_angle_deg)
    tooth_half_width = pitch / 4.0

    for i in range(teeth_count):
        x_center = pitch * (i + 0.5)

        # 歯底（歯元）
        profile_points.append((x_center - tooth_half_width * 1.3, base_height))
        # 立ち上がり歯面
        profile_points.append((x_center - tooth_half_width * 0.5, base_height + tooth_height))
        # 歯先
        profile_points.append((x_center + tooth_half_width * 0.5, base_height + tooth_height))
        # 下り歯面
        profile_points.append((x_center + tooth_half_width * 1.3, base_height))

    # プロファイルを閉じる
    profile_points.append((0, base_height))

    # 押し出し
    holes = []

    # 指定時に取付穴を追加
    if mounting_holes:
        hole_radius = hole_diameter_mm / 2.0
        hole_y = base_height / 2.0
        hole_spacing = length / (teeth_count // 5 + 1)

        for i in range(1, teeth_count // 5 + 1):
            hole_x = hole_spacing * i
            holes.append(_circle_points((hole_x, hole_y), hole_radius, segments=32))

    mesh = _extrude_profile(profile_points, holes, face_width_mm)

    if mesh is None:
        # 簡易直方体にフォールバック
        mesh = trimesh.creation.box(extents=(length, height_mm, face_width_mm))
        mesh.apply_translation((length / 2.0, height_mm / 2.0, face_width_mm / 2.0))

    return mesh
