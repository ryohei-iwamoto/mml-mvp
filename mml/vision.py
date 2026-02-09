import cv2
import numpy as np


def _to_gray(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _preprocess(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(clean, 50, 150)
    return clean, edges


def _resample_points(points, max_points=1200):
    if len(points) <= max_points:
        return points
    step = len(points) / float(max_points)
    return [points[int(i * step)] for i in range(max_points)]


def _find_outline(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    points = [[int(p[0][0]), int(p[0][1])] for p in largest]
    if len(points) < 3:
        return None
    return _resample_points(points)


def _find_holes(gray, binary):
    # 1) 外部輪郭による穴検出（従来方式）
    contours_ext, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if not contours_ext:
        contours_ext = []
    max_area = max((cv2.contourArea(c) for c in contours_ext), default=0)
    for c in contours_ext:
        area = cv2.contourArea(c)
        if max_area > 0 and area >= max_area * 0.9:
            continue
        if area < 20:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * area / (peri * peri)
        if circularity < 0.65:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        holes.append(
            {
                "center_px": [float(x), float(y)],
                "radius_px": float(r),
                "confidence": float(min(1.0, circularity)),
            }
        )
    if holes:
        return holes

    # 2) 輪郭階層による内部穴検出（最大輪郭の全子孫から円形を探す）
    contours_tree, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours_tree and hierarchy is not None:
        largest_idx = max(range(len(contours_tree)), key=lambda i: cv2.contourArea(contours_tree[i]))
        largest_area = cv2.contourArea(contours_tree[largest_idx])
        # 最大輪郭の全子孫を収集
        descendants = set()
        queue = []
        # 直接の子を探す
        for i in range(len(contours_tree)):
            if hierarchy[0][i][3] == largest_idx:
                queue.append(i)
                descendants.add(i)
        # 子孫を再帰的に収集
        while queue:
            parent = queue.pop(0)
            for i in range(len(contours_tree)):
                if hierarchy[0][i][3] == parent and i not in descendants:
                    descendants.add(i)
                    queue.append(i)
        for i in descendants:
            area = cv2.contourArea(contours_tree[i])
            if area < 20:
                continue
            # 親の面積の50%を超える大きな輪郭は穴ではない
            if largest_area > 0 and area > largest_area * 0.5:
                continue
            peri = cv2.arcLength(contours_tree[i], True)
            if peri == 0:
                continue
            circularity = 4 * np.pi * area / (peri * peri)
            if circularity < 0.65:
                continue
            (x, y), r = cv2.minEnclosingCircle(contours_tree[i])
            holes.append(
                {
                    "center_px": [float(x), float(y)],
                    "radius_px": float(r),
                    "confidence": float(min(1.0, circularity)),
                }
            )
        # 重複を除去（近接する穴をマージ）
        if holes:
            merged = []
            used = [False] * len(holes)
            for i, h in enumerate(holes):
                if used[i]:
                    continue
                best = h
                for j in range(i + 1, len(holes)):
                    if used[j]:
                        continue
                    dx = h["center_px"][0] - holes[j]["center_px"][0]
                    dy = h["center_px"][1] - holes[j]["center_px"][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < max(h["radius_px"], holes[j]["radius_px"]):
                        used[j] = True
                        if holes[j]["confidence"] > best["confidence"]:
                            best = holes[j]
                merged.append(best)
            holes = merged
    if holes:
        return holes

    # 3) Hough円検出（フォールバック、制限付き）
    min_dim = min(gray.shape[:2])
    max_radius = max(int(min_dim * 0.25), 20)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=15,
        param1=100,
        param2=25,
        minRadius=4,
        maxRadius=max_radius,
    )
    if circles is None:
        return []

    # 外形輪郭の内部にある円のみ保持
    outline_cnt = None
    if contours_ext:
        outline_cnt = max(contours_ext, key=cv2.contourArea)

    for (x, y, r) in circles[0]:
        if outline_cnt is not None:
            dist = cv2.pointPolygonTest(outline_cnt, (float(x), float(y)), True)
            if dist < r * 0.5:
                continue
        holes.append(
            {
                "center_px": [float(x), float(y)],
                "radius_px": float(r),
                "confidence": 0.6,
            }
        )

    # 最大50個に制限
    if len(holes) > 50:
        holes.sort(key=lambda h: h["confidence"], reverse=True)
        holes = holes[:50]

    return holes


def _find_bend_lines(edges, outline_points):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=40, maxLineGap=5)
    if lines is None:
        return []
    bend_lines = []
    outline_cnt = None
    if outline_points:
        outline_cnt = np.array(outline_points, dtype=np.int32)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 50:
            continue
        if outline_cnt is not None:
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            inside = cv2.pointPolygonTest(outline_cnt, (mx, my), False) >= 0
            if not inside:
                continue
        conf = min(1.0, length / 200.0)
        bend_lines.append(
            {"line_px": [[int(x1), int(y1)], [int(x2), int(y2)]], "confidence": float(conf)}
        )
    return bend_lines


def run_vision(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to read image")
    gray = _to_gray(img)
    binary, edges = _preprocess(gray)
    outline_points = _find_outline(binary)
    holes = _find_holes(gray, binary)
    bend_lines = _find_bend_lines(edges, outline_points)
    outline_type = "spline" if outline_points and len(outline_points) > 20 else "polygon"
    result = {
        "outline": {"type": outline_type, "points_px": outline_points or []},
        "holes": holes,
        "bend_lines": bend_lines,
        "notes_regions": [],
    }
    return result


def normalize_vision(raw):
    if raw is None:
        return {"outline": {"type": "polygon", "points_px": []}, "holes": [], "bend_lines": [], "notes_regions": []}

    outline = raw.get("outline", {})
    outline_type = outline.get("type") or raw.get("outline_type")
    points_px = outline.get("points_px")
    if points_px is None:
        points_px = outline.get("coordinates") or outline.get("points") or []
    if not outline_type:
        outline_type = "spline" if points_px and len(points_px) > 20 else "polygon"

    holes = []
    for h in raw.get("holes", []) or []:
        if not isinstance(h, dict):
            continue
        center = h.get("center_px") or h.get("center") or h.get("center_xy")
        if center is None and h.get("cx") is not None and h.get("cy") is not None:
            center = [h.get("cx"), h.get("cy")]
        radius = h.get("radius_px") or h.get("radius")
        if center is None or radius is None:
            continue
        holes.append(
            {
                "center_px": [float(center[0]), float(center[1])],
                "radius_px": float(radius),
                "confidence": float(h.get("confidence", 0.6)),
            }
        )

    bend_lines = []
    for b in raw.get("bend_lines", []) or []:
        if not isinstance(b, dict):
            continue
        line_px = b.get("line_px")
        if line_px is None:
            start = b.get("start")
            end = b.get("end")
            if start is not None and end is not None:
                line_px = [start, end]
        if not line_px or len(line_px) != 2:
            continue
        bend_lines.append(
            {
                "line_px": [
                    [float(line_px[0][0]), float(line_px[0][1])],
                    [float(line_px[1][0]), float(line_px[1][1])],
                ],
                "confidence": float(b.get("confidence", 0.6)),
            }
        )

    return {
        "outline": {"type": outline_type, "points_px": points_px or []},
        "holes": holes,
        "bend_lines": bend_lines,
        "notes_regions": raw.get("notes_regions", []) or [],
        "part_hint": raw.get("part_hint"),
        "part_hint_confidence": raw.get("part_hint_confidence"),
    }
