import cv2
import ezdxf
import numpy as np


def _ensure_linetype(doc, name, description, pattern):
    if name in doc.linetypes:
        return
    doc.linetypes.new(name, dxfattribs={"description": description, "pattern": pattern})


def _ensure_layer(doc, name, color=7, linetype="Continuous"):
    if name in doc.layers:
        layer = doc.layers.get(name)
        layer.dxf.color = color
        layer.dxf.linetype = linetype
        return
    doc.layers.new(name, dxfattribs={"color": color, "linetype": linetype})


def _extract_thickness_mm(mml):
    for c in mml.get("constraints", []):
        if c.get("kind") == "min_thickness" and c.get("value_mm") is not None:
            return float(c.get("value_mm"))
    if mml.get("thickness_mm") is not None:
        return float(mml.get("thickness_mm"))
    return 5.0


def _bounds(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _draw_outline(msp, outline, layer):
    if not outline:
        return
    if len(outline) < 3:
        return
    msp.add_lwpolyline(outline, close=True, dxfattribs={"layer": layer})


def _translate_points(points, dx, dy):
    return [[p[0] + dx, p[1] + dy] for p in points]


def _add_text_line(msp, text, x, y, layer="TEXT", height=3.0):
    msp.add_text(
        text,
        dxfattribs={
            "layer": layer,
            "height": height,
            "insert": (x, y),
        },
    )


def draw_dxf(mml, out_path):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4  # millimeters
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()

    _ensure_linetype(doc, "CENTER", "Center ____ _ ____ _ ____", [0.0, 10.0, -2.0, 2.0, -2.0])
    _ensure_linetype(doc, "HIDDEN", "Hidden __ __ __ __", [0.0, 6.0, -3.0])
    _ensure_layer(doc, "OUTLINE", color=7, linetype="Continuous")
    _ensure_layer(doc, "HOLES", color=7, linetype="Continuous")
    _ensure_layer(doc, "BEND", color=2, linetype="CENTER")
    _ensure_layer(doc, "CENTER", color=3, linetype="CENTER")
    _ensure_layer(doc, "HIDDEN", color=8, linetype="HIDDEN")
    _ensure_layer(doc, "TEXT", color=7, linetype="Continuous")
    _ensure_layer(doc, "VIEW_FRAME", color=8, linetype="Continuous")

    outline_data = mml.get("geometry", {}).get("outline", {})
    outline = outline_data.get("points_mm", [])
    holes = mml.get("geometry", {}).get("holes", [])
    bend = mml.get("geometry", {}).get("bend")

    if outline:
        min_x, min_y, max_x, max_y = _bounds(outline)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 100.0, 60.0

    width = max(1.0, max_x - min_x)
    depth = max(1.0, max_y - min_y)
    thickness = _extract_thickness_mm(mml)
    gap = max(20.0, 0.25 * max(width, depth))

    # Third-angle projection layout:
    # Front at origin, Top above front, Right view right of front.
    front_origin = (0.0, 0.0)
    top_origin = (0.0, thickness + gap)
    right_origin = (width + gap, 0.0)

    top_outline = _translate_points(outline, -min_x + top_origin[0], -min_y + top_origin[1])
    _draw_outline(msp, top_outline, "OUTLINE")

    # Top view holes and centerlines
    for h in holes:
        center = h.get("center_mm")
        dia = h.get("diameter_mm")
        if not center or not dia:
            continue
        cx = center[0] - min_x + top_origin[0]
        cy = center[1] - min_y + top_origin[1]
        r = dia / 2.0
        msp.add_circle((cx, cy), r, dxfattribs={"layer": "HOLES"})
        msp.add_line((cx - r - 3.0, cy), (cx + r + 3.0, cy), dxfattribs={"layer": "CENTER"})
        msp.add_line((cx, cy - r - 3.0), (cx, cy + r + 3.0), dxfattribs={"layer": "CENTER"})

    if bend:
        line = bend.get("line_mm")
        if line and len(line) == 2:
            p1 = (line[0][0] - min_x + top_origin[0], line[0][1] - min_y + top_origin[1])
            p2 = (line[1][0] - min_x + top_origin[0], line[1][1] - min_y + top_origin[1])
            msp.add_line(p1, p2, dxfattribs={"layer": "BEND"})

    # Front view (X-Z)
    fx0, fy0 = front_origin
    front_rect = [(fx0, fy0), (fx0 + width, fy0), (fx0 + width, fy0 + thickness), (fx0, fy0 + thickness)]
    _draw_outline(msp, front_rect, "OUTLINE")

    # Hidden lines for through holes in front view
    for h in holes:
        center = h.get("center_mm")
        dia = h.get("diameter_mm")
        if not center or not dia:
            continue
        cx = center[0] - min_x
        r = dia / 2.0
        msp.add_line((fx0 + cx - r, fy0), (fx0 + cx - r, fy0 + thickness), dxfattribs={"layer": "HIDDEN"})
        msp.add_line((fx0 + cx + r, fy0), (fx0 + cx + r, fy0 + thickness), dxfattribs={"layer": "HIDDEN"})

    # Right view (Y-Z)
    rx0, ry0 = right_origin
    right_rect = [(rx0, ry0), (rx0 + depth, ry0), (rx0 + depth, ry0 + thickness), (rx0, ry0 + thickness)]
    _draw_outline(msp, right_rect, "OUTLINE")

    # Hidden lines for through holes in right view
    for h in holes:
        center = h.get("center_mm")
        dia = h.get("diameter_mm")
        if not center or not dia:
            continue
        cy = center[1] - min_y
        r = dia / 2.0
        msp.add_line((rx0 + cy - r, ry0), (rx0 + cy - r, ry0 + thickness), dxfattribs={"layer": "HIDDEN"})
        msp.add_line((rx0 + cy + r, ry0), (rx0 + cy + r, ry0 + thickness), dxfattribs={"layer": "HIDDEN"})

    # Lightweight view frames improve readability in common CAD viewers.
    top_frame = [
        (top_origin[0], top_origin[1]),
        (top_origin[0] + width, top_origin[1]),
        (top_origin[0] + width, top_origin[1] + depth),
        (top_origin[0], top_origin[1] + depth),
    ]
    _draw_outline(msp, top_frame, "VIEW_FRAME")
    _draw_outline(msp, front_rect, "VIEW_FRAME")
    _draw_outline(msp, right_rect, "VIEW_FRAME")

    _add_text_line(msp, "TOP VIEW", top_origin[0], top_origin[1] + depth + 6.0)
    _add_text_line(msp, "FRONT VIEW", fx0, fy0 + thickness + 6.0)
    _add_text_line(msp, "RIGHT VIEW", rx0, ry0 + thickness + 6.0)

    # Basic dimensions/metadata text block
    holes = mml.get("geometry", {}).get("holes", [])
    part = mml.get("part")
    material = mml.get("material", {}).get("name")

    hole_std = None
    if holes:
        hole_std = holes[0].get("standard")

    bend_info = ""
    if bend:
        bend_info = f"BEND: {bend.get('angle_deg')}deg R={bend.get('inner_radius_mm')}"

    lines = [
        f"PART: {part}",
        f"MAT: {material} t={thickness}",
        f"SIZE: W={width:.2f} D={depth:.2f} T={thickness:.2f}",
        f"HOLES: {len(holes)}x {hole_std}",
        bend_info,
    ]
    text_x = right_origin[0] + depth + gap * 0.5
    text_y = top_origin[1] + depth
    for i, line in enumerate([l for l in lines if l]):
        _add_text_line(msp, line, text_x, text_y - 5.0 * i)

    doc.saveas(out_path)


def _collect_bounds(mml):
    points = []
    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    points.extend(outline)
    for h in mml.get("geometry", {}).get("holes", []):
        center = h.get("center_mm")
        radius = h.get("diameter_mm")
        if center and radius:
            r = radius / 2.0
            points.extend(
                [
                    [center[0] - r, center[1] - r],
                    [center[0] + r, center[1] + r],
                ]
            )
    bend = mml.get("geometry", {}).get("bend")
    if bend:
        line = bend.get("line_mm") or []
        points.extend(line)
    if not points:
        return 0, 0, 100, 100
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def draw_png(mml, out_path, scale=None):
    min_x, min_y, max_x, max_y = _collect_bounds(mml)
    width_mm = max_x - min_x
    height_mm = max_y - min_y
    if width_mm <= 0:
        width_mm = 1.0
    if height_mm <= 0:
        height_mm = 1.0

    if scale is None:
        target_px = 1200
        scale = max(1.0, min(6.0, target_px / max(width_mm, height_mm)))

    margin = 20
    width_px = int(round(width_mm * scale + margin * 2))
    height_px = int(round(height_mm * scale + margin * 2))

    canvas = np.full((height_px, width_px, 3), 255, dtype=np.uint8)

    def to_px(pt):
        x = int(round((pt[0] - min_x) * scale + margin))
        y = int(round((max_y - pt[1]) * scale + margin))
        return x, y

    outline = mml.get("geometry", {}).get("outline", {}).get("points_mm", [])
    if outline:
        pts = np.array([to_px(p) for p in outline], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, (0, 0, 0), 2)

    for h in mml.get("geometry", {}).get("holes", []):
        center = h.get("center_mm")
        diameter = h.get("diameter_mm")
        if center and diameter:
            cx, cy = to_px(center)
            radius_px = int(round((diameter / 2.0) * scale))
            if radius_px > 0:
                cv2.circle(canvas, (cx, cy), radius_px, (0, 0, 0), 2)

    bend = mml.get("geometry", {}).get("bend")
    if bend:
        line = bend.get("line_mm")
        if line and len(line) == 2:
            p1 = to_px(line[0])
            p2 = to_px(line[1])
            cv2.line(canvas, p1, p2, (0, 0, 0), 1)

    # 最小限の形状でも空出力にならないよう簡易注釈を付与。
    part = str(mml.get("part") or "Part")
    holes = mml.get("geometry", {}).get("holes", [])
    text_lines = [
        f"PART: {part}",
        f"HOLES: {len(holes)}",
    ]
    for idx, text in enumerate(text_lines):
        cv2.putText(
            canvas,
            text,
            (margin, height_px - margin - 10 - idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    if not outline and not holes and not mml.get("geometry", {}).get("bend"):
        cv2.putText(
            canvas,
            "No geometry",
            (margin, margin + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        raise ValueError("PNG encoding failed")
    with open(out_path, "wb") as f:
        f.write(buffer.tobytes())
