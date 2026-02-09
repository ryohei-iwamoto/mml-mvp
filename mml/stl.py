import math

from shapely.geometry import Polygon
import trimesh


def _circle_points(center, radius, segments=48):
    cx, cy = center
    pts = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts


def _gear_outline(outer_diameter, teeth_count):
    teeth = max(8, int(teeth_count))
    outer_r = float(outer_diameter) / 2.0
    root_r = outer_r * 0.85
    points = []
    total = teeth * 2
    for i in range(total):
        theta = 2.0 * math.pi * i / total
        r = outer_r if i % 2 == 0 else root_r
        points.append((r * math.cos(theta), r * math.sin(theta)))
    return points


def _arc_points(center, radius, start_deg, end_deg, segments=12):
    cx, cy = center
    pts = []
    for i in range(segments + 1):
        t = i / float(segments)
        deg = start_deg + (end_deg - start_deg) * t
        theta = math.radians(deg)
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        pts.append((x, y))
    return pts


def _rounded_rect_outline(width, height, radius, segments=8):
    w = float(width)
    h = float(height)
    r = min(float(radius), w / 2.0, h / 2.0)
    if r <= 0:
        return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    pts = []
    pts.extend(_arc_points((w - r, r), r, -90, 0, segments))
    pts.extend(_arc_points((w - r, h - r), r, 0, 90, segments))
    pts.extend(_arc_points((r, h - r), r, 90, 180, segments))
    pts.extend(_arc_points((r, r), r, 180, 270, segments))
    return pts


def _thickness_from_mml(mml):
    for c in mml.get("constraints", []):
        if c.get("kind") == "min_thickness":
            value = c.get("value_mm")
            if value is not None:
                return float(value)
    return 5.0


def _scale_from_mml(mml):
    intent = (mml or {}).get("intent", {}) or {}
    cfg = intent.get("arm_config") or {}
    try:
        reach = float(cfg.get("reach_mm", 300.0))
    except (TypeError, ValueError):
        reach = 300.0
    return max(0.6, min(1.4, reach / 300.0))


def _arm_dims(mml, scale):
    intent = (mml or {}).get("intent", {}) or {}
    dims = intent.get("arm_dims") or {}
    def _pick(key, default):
        try:
            return float(dims.get(key, default))
        except (TypeError, ValueError):
            return float(default)
    return {
        "link_length_mm": _pick("link_length_mm", 160.0 * scale),
        "link_width_mm": _pick("link_width_mm", 30.0 * scale),
        "link_hole_offset_mm": _pick("link_hole_offset_mm", 18.0 * scale),
        "joint_outer_diameter_mm": _pick("joint_outer_diameter_mm", 56.0 * scale),
        "joint_hole_diameter_mm": _pick("joint_hole_diameter_mm", 12.0 * scale),
        "gear_outer_diameter_mm": _pick("gear_outer_diameter_mm", 60.0 * scale),
        "gear_bore_diameter_mm": _pick("gear_bore_diameter_mm", 12.0 * scale),
        "shaft_diameter_mm": _pick("shaft_diameter_mm", 12.0 * scale),
        "bearing_outer_diameter_mm": _pick("bearing_outer_diameter_mm", 40.0 * scale),
        "bearing_inner_diameter_mm": _pick("bearing_inner_diameter_mm", 16.0 * scale),
        "motor_outer_diameter_mm": _pick("motor_outer_diameter_mm", 50.0 * scale),
        "motor_mount_width_mm": _pick("motor_mount_width_mm", 80.0 * scale),
        "motor_mount_height_mm": _pick("motor_mount_height_mm", 60.0 * scale),
        "base_width_mm": _pick("base_width_mm", 120.0 * scale),
        "base_height_mm": _pick("base_height_mm", 90.0 * scale),
    }


def _extrude_profile(outline, holes, height, z_offset=0.0):
    poly = Polygon(outline, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    mesh = trimesh.creation.extrude_polygon(poly, height=height, triangulate_kwargs={"engine": "earcut"})
    if z_offset:
        mesh.apply_translation((0.0, 0.0, z_offset))
    return mesh


def _link_profile():
    length = 160.0
    width = 30.0
    fillet = 6.0
    outline = _rounded_rect_outline(length, width, fillet, segments=10)
    hole_d = 8.0
    hole_r = hole_d / 2.0
    holes = [
        _circle_points((18.0, width / 2.0), hole_r, segments=48),
        _circle_points((length - 18.0, width / 2.0), hole_r, segments=48),
    ]
    return outline, holes, hole_r


def _joint_profile():
    outer_r = 28.0
    shaft_r = 6.0
    outline = _circle_points((0.0, 0.0), outer_r, segments=80)
    holes = [_circle_points((0.0, 0.0), shaft_r, segments=48)]
    return outline, holes, shaft_r


def _base_profile():
    width = 120.0
    height = 90.0
    fillet = 8.0
    outline = _rounded_rect_outline(width, height, fillet, segments=10)
    hole_r = 4.0
    offset = 14.0
    holes = [
        _circle_points((offset, offset), hole_r, segments=36),
        _circle_points((width - offset, offset), hole_r, segments=36),
        _circle_points((width - offset, height - offset), hole_r, segments=36),
        _circle_points((offset, height - offset), hole_r, segments=36),
    ]
    return outline, holes, hole_r


def _primitive_for_part(part_name, thickness, mml=None):
    name = (part_name or "").lower()
    scale = _scale_from_mml(mml)
    dims = _arm_dims(mml, scale)
    meshes = []

    def _scale_points(points):
        return [(p[0] * scale, p[1] * scale) for p in points]

    def _scale_holes(holes):
        return [[(p[0] * scale, p[1] * scale) for p in hole] for hole in holes]
    if "link" in name or "arm" in name:
        length = dims["link_length_mm"]
        width = dims["link_width_mm"]
        offset = dims["link_hole_offset_mm"]
        fillet = min(6.0 * scale, width / 2.0)
        outline = _rounded_rect_outline(length, width, fillet, segments=10)
        hole_r = dims["joint_hole_diameter_mm"] / 2.0
        holes = [
            _circle_points((offset, width / 2.0), hole_r, segments=48),
            _circle_points((length - offset, width / 2.0), hole_r, segments=48),
        ]
        base = _extrude_profile(outline, holes, thickness)
        if base:
            meshes.append(base)
        # 穴周りのボスリング（簡易段付き形状）。
        boss_height = max(2.0, thickness * 0.4)
        boss_outer = hole_r + 4.0
        for center in [(offset, width / 2.0), (length - offset, width / 2.0)]:
            ring_outline = _circle_points(center, boss_outer, segments=48)
            ring_hole = [_circle_points(center, hole_r, segments=48)]
            ring = _extrude_profile(ring_outline, ring_hole, boss_height, z_offset=thickness)
            if ring:
                meshes.append(ring)
    elif "joint" in name:
        outer_r = dims["joint_outer_diameter_mm"] / 2.0
        shaft_r = dims["joint_hole_diameter_mm"] / 2.0
        outline = _circle_points((0.0, 0.0), outer_r, segments=80)
        holes = [_circle_points((0.0, 0.0), shaft_r, segments=48)]
        base = _extrude_profile(outline, holes, thickness)
        if base:
            meshes.append(base)
        # 軸周りのボス（段付きカラー）。
        collar_outer = shaft_r + 6.0
        collar_height = max(3.0, thickness * 0.5)
        ring_outline = _circle_points((0.0, 0.0), collar_outer, segments=72)
        ring_hole = [_circle_points((0.0, 0.0), shaft_r, segments=48)]
        ring = _extrude_profile(ring_outline, ring_hole, collar_height, z_offset=thickness)
        if ring:
            meshes.append(ring)
    elif "base" in name:
        width = dims["base_width_mm"]
        height = dims["base_height_mm"]
        fillet = min(8.0 * scale, width / 2.0, height / 2.0)
        outline = _rounded_rect_outline(width, height, fillet, segments=10)
        hole_r = dims["joint_hole_diameter_mm"] / 2.0
        offset = min(14.0 * scale, width / 4.0, height / 4.0)
        holes = [
            _circle_points((offset, offset), hole_r, segments=36),
            _circle_points((width - offset, offset), hole_r, segments=36),
            _circle_points((width - offset, height - offset), hole_r, segments=36),
            _circle_points((offset, height - offset), hole_r, segments=36),
        ]
        base = _extrude_profile(outline, holes, thickness)
        if base:
            meshes.append(base)
        # 取付穴のスタンドオフ。
        stand_height = max(3.0, thickness * 0.6)
        stand_outer = hole_r + 3.0
        offsets = [
            (offset, offset),
            (width - offset, offset),
            (width - offset, height - offset),
            (offset, height - offset),
        ]
        for center in offsets:
            ring_outline = _circle_points(center, stand_outer, segments=36)
            ring_hole = [_circle_points(center, hole_r, segments=36)]
            ring = _extrude_profile(ring_outline, ring_hole, stand_height, z_offset=thickness)
            if ring:
                meshes.append(ring)
    elif "end_effector" in name or "gripper" in name:
        return trimesh.creation.box(extents=(50.0 * scale, 30.0 * scale, thickness))
    elif "shaft" in name:
        return trimesh.creation.cylinder(
            radius=dims["shaft_diameter_mm"] / 2.0, height=80.0 * scale, sections=48
        )
    elif "rotor" in name or "stator" in name or "motor" in name or "actuator" in name:
        return trimesh.creation.cylinder(
            radius=dims["motor_outer_diameter_mm"] / 2.0, height=40.0 * scale, sections=64
        )
    elif "motor_mount" in name or "mount" in name:
        outline = _rounded_rect_outline(
            dims["motor_mount_width_mm"], dims["motor_mount_height_mm"], 6.0 * scale, segments=10
        )
        holes = [
            _circle_points((12.0 * scale, 12.0 * scale), 3.0 * scale, segments=36),
            _circle_points((dims["motor_mount_width_mm"] - 12.0 * scale, 12.0 * scale), 3.0 * scale, segments=36),
            _circle_points(
                (dims["motor_mount_width_mm"] - 12.0 * scale, dims["motor_mount_height_mm"] - 12.0 * scale),
                3.0 * scale,
                segments=36,
            ),
            _circle_points((12.0 * scale, dims["motor_mount_height_mm"] - 12.0 * scale), 3.0 * scale, segments=36),
            _circle_points(
                (dims["motor_mount_width_mm"] / 2.0, dims["motor_mount_height_mm"] / 2.0),
                6.0 * scale,
                segments=48,
            ),
        ]
        mesh = _extrude_profile(outline, holes, thickness)
        if mesh:
            return mesh
    elif "bearing" in name:
        outline = _circle_points((0.0, 0.0), dims["bearing_outer_diameter_mm"] / 2.0, segments=96)
        holes = [_circle_points((0.0, 0.0), dims["bearing_inner_diameter_mm"] / 2.0, segments=64)]
        mesh = _extrude_profile(outline, holes, thickness)
        if mesh:
            return mesh
    elif "spacer" in name:
        outline = _circle_points((0.0, 0.0), 15.0 * scale, segments=64)
        holes = [_circle_points((0.0, 0.0), dims["shaft_diameter_mm"] / 2.0, segments=48)]
        mesh = _extrude_profile(outline, holes, thickness)
        if mesh:
            return mesh
    elif "bracket" in name:
        outline = [(0.0, 0.0), (60.0, 0.0), (60.0, 15.0), (20.0, 15.0), (20.0, 50.0), (0.0, 50.0)]
        holes = [
            _circle_points((10.0, 10.0), 3.0, segments=36),
            _circle_points((10.0, 40.0), 3.0, segments=36),
        ]
        mesh = _extrude_profile(outline, holes, thickness)
        if mesh:
            return mesh
    elif "gear" in name:
        outline = _gear_outline(dims["gear_outer_diameter_mm"], 24)
        mesh = _extrude_profile(outline, [], thickness)
        if mesh:
            return mesh
        return trimesh.creation.cylinder(
            radius=dims["gear_outer_diameter_mm"] / 2.0, height=thickness, sections=96
        )
    else:
        return trimesh.creation.box(extents=(60.0 * scale, 40.0 * scale, thickness))

    if not meshes:
        return trimesh.creation.box(extents=(60.0 * scale, 40.0 * scale, thickness))
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def write_stl(mml, out_path):
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    thickness = _thickness_from_mml(mml)
    if not outline:
        mesh = _primitive_for_part(mml.get("part"), thickness, mml=mml)
        mesh.export(out_path)
        return True

    holes = []
    for h in mml.get("geometry", {}).get("holes", []):
        center = h.get("center_mm")
        diameter = h.get("diameter_mm")
        if center and diameter:
            holes.append(_circle_points(center, float(diameter) / 2.0, segments=32))

    poly = Polygon(outline, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return False

    mesh = trimesh.creation.extrude_polygon(poly, height=thickness, triangulate_kwargs={"engine": "earcut"})

    part_name = (mml.get("part") or "").lower()
    holes_mm = mml.get("geometry", {}).get("holes", []) or []
    bosses = []
    if holes_mm and ("link" in part_name or "arm" in part_name or "base" in part_name):
        boss_height = max(2.0, thickness * 0.4)
        for h in holes_mm:
            center = h.get("center_mm")
            dia = h.get("diameter_mm")
            if not center or not dia:
                continue
            hole_r = float(dia) / 2.0
            boss_outer = hole_r + 4.0
            ring_outline = _circle_points(center, boss_outer, segments=48)
            ring_hole = [_circle_points(center, hole_r, segments=48)]
            ring = _extrude_profile(ring_outline, ring_hole, boss_height, z_offset=thickness)
            if ring:
                bosses.append(ring)
    elif holes_mm and "joint" in part_name:
        h = holes_mm[0]
        center = h.get("center_mm")
        dia = h.get("diameter_mm")
        if center and dia:
            hole_r = float(dia) / 2.0
            collar_outer = hole_r + 6.0
            collar_height = max(3.0, thickness * 0.5)
            ring_outline = _circle_points(center, collar_outer, segments=72)
            ring_hole = [_circle_points(center, hole_r, segments=48)]
            ring = _extrude_profile(ring_outline, ring_hole, collar_height, z_offset=thickness)
            if ring:
                bosses.append(ring)

    if bosses:
        mesh = trimesh.util.concatenate([mesh] + bosses)

    mesh.export(out_path)
    return True
