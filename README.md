# MML-MVP: Mechanical Modeling Language (Graduation Research MVP)

## 0. Why / Concept (やりたいことの概念)

機械設計では、最終成果物として「寸法が確定した図面/CAD」が作られる。
しかし設計の初期段階では、実際に重要なのは以下である。

- どんな動き・機能を実現したいのか（意図）
- そのためにどんな機構を選ぶのか（構造）
- どんな制約を満たす必要があるのか（加工・規格・強度・組立）
- 材料や製法が変わっても成立する抽象性（柔軟性）

図面に落とし込むと寸法が固定されるため、材料や製法の変更に追従しづらい。
そこで本研究では、ラフスケッチと対話入力から「設計意図」を抽象言語として保持し、
その抽象モデルから図面を生成できるパイプラインを作る。

本MVPはその第一段階として、板金ブラケット/プレート系に対象を限定し、
以下を一気通貫で実現する：

**ラフ画像 → 認識 → 対話で曖昧さ解消 → 中間言語(MML-JSON) → 図面(DXF)出力**

---

## 1. What is "Mechanical Modeling Language" in this MVP

このMVPにおける「機械モデリング言語」は、CADのように形状操作を列挙する言語ではない。
設計の意図を、次のレイヤで統合表現するための中間言語である。

- Feature: 何を作るか（例: plate, bracket）
- Interface: 何と繋ぐか（例: bolt pattern, hole standard）
- Constraint: 何を満たすべきか（例: 板厚 >= 2mm, 曲げR >= t, 端距離 >= 2d）
- Provenance: どれが画像推定で、どれが対話で確定したか（追跡可能性）

MVPでは中間言語として **MML-JSON** を採用する。
将来的にテキストDSLへ拡張可能だが、まずは自動生成・検証しやすいJSONに固定する。

---

## 2. Scope (MVPでやる / やらない)

### ✅ MVPでやる
- 白背景・黒線のラフ画像からの要素認識（外形/穴/曲げ線候補）
- 認識結果が曖昧な箇所を「対話」で確定（スケール、穴規格など）
- 中間言語(MML-JSON)の生成
- 図面(DXF)の生成（外形線・穴・曲げ線・注記）
- 推論レポート(report.json)出力（確信度、質問、回答、確定値）

### ❌ MVPでやらない
- 3D STEP生成
- 本格FEM/最適化（強度計算は将来拡張）
- 写真や複雑背景のラフ対応
- 完全自動で寸法が確定すること（必ず1つ以上ユーザー回答でスケール確定）

---

## 3. High-level Pipeline (全体パイプライン)

1) `vision`: 画像から外形・穴・曲げ線候補を抽出し、`vision.json`へ
2) `interact`: `vision.json`を元に、スケールや穴規格などの不足情報を質問して確定
3) `emit`: 確定した内容を `mml.json` (MML-JSON) と `report.json` として出力
4) `draw`: `mml.json` から図面 `drawing.dxf` を生成

---

## 4. CLI

- `mml vision input.png -o out/`
  - outputs: `out/vision.json`

- `mml interact input.png --chat=rule -o out/`
  - outputs: `out/mml.json`, `out/report.json`

- `mml draw out/mml.json -o out/`
  - outputs: `out/drawing.dxf`

- `mml pipeline input.png --chat=rule -o out/`
  - outputs: `out/vision.json`, `out/mml.json`, `out/report.json`, `out/drawing.dxf`

---

## 5. MML-JSON (Intermediate Language) Schema (MVP minimal)

```json
{
  "part": "Bracket",
  "units": "mm",
  "scale": { "px_to_mm": 0.5 },
  "material": { "name": "A5052" },
  "process": { "name": "sheet_metal" },
  "geometry": {
    "outline": { "type": "polygon", "points_mm": [[0,0],[100,0],[100,50],[0,50]] },
    "holes": [
      { "type": "clearance", "standard": "M5", "diameter_mm": 5.5, "center_mm": [50,25] }
    ],
    "bend": {
      "line_mm": [[60,0],[60,50]],
      "angle_deg": 90,
      "inner_radius_mm": 2.0
    }
  },
  "constraints": [
    { "kind": "min_thickness", "value_mm": 2.0 },
    { "kind": "bend_radius_gte_thickness" },
    { "kind": "edge_distance_gte", "multiplier": 2.0 }
  ],
  "provenance": {
    "vision": { "file": "input.png", "version": "0.1" },
    "chat": { "mode": "rule", "questions": [], "answers": [] }
  }
}
```

## 6. Acceptance Criteria (完了条件)

 画像→認識→対話→中間言語→DXF出力が一気通貫で動く

 mml.json に provenance（vision/chat）が含まれる

 DXFがレイヤ分けされ、外形と穴が円で出力される

 自動テストで3ケースが再現可能（画像はテスト内で生成）

---

## 7. Vision Module Specification (Image Recognition)

### 7.1 Assumptions (MVP Constraints)
- Input image: white background, black line drawing
- Single part per image
- Target domain: plate / sheet-metal bracket
- No perspective distortion
- No filled regions (line art only)

### 7.2 Preprocessing
1. Convert to grayscale
2. Binarize (Otsu threshold)
3. Morphological open/close to remove noise
4. Edge detection (Canny)

### 7.3 Feature Extraction

#### Outline Detection
- Extract all contours
- Select the contour with the largest area as the outer outline
- Approximate polygon (Douglas–Peucker)
- Store as ordered polygon in pixel coordinates

#### Hole Detection
- Use Hough Circle Transform OR
- Detect closed contours with high circularity
- Store center (cx, cy), radius (r), confidence

#### Bend Line Detection
- Use probabilistic Hough line detection
- Filter long straight lines inside the outline
- Store line endpoints and confidence

### 7.4 Vision Output Format (`vision.json`)
```json
{
  "outline": { "type": "polygon", "points_px": [[...]] },
  "holes": [
    { "center_px": [120, 80], "radius_px": 9, "confidence": 0.82 }
  ],
  "bend_lines": [
    { "line_px": [[60, 10], [60, 180]], "confidence": 0.63 }
  ],
  "notes_regions": [
    { "bbox_px": [x,y,w,h], "confidence": 0.55 }
  ]
}

```

8. Interaction / Chat Module Specification
8.1 Purpose

Resolve ambiguity that cannot be determined from image recognition alone.

Typical ambiguities:

Pixel-to-mm scale

Hole standard (M4 / M5 / M6 ...)

Plate thickness

Bend angle and inner radius

8.2 Mandatory Questions

At least one scale-defining question must be answered.

Examples:

"What is the real width (mm) of the outer plate?"

"What bolt standard are these holes for?"

8.3 Rule-based Chat (MVP Default)

For MVP, a deterministic rule-based question generator is sufficient.

Example logic:

If scale.px_to_mm undefined → ask for one reference dimension

If hole radii vary → ask if they should be unified

If bend detected and angle undefined → ask bend angle (default suggestion: 90deg)

8.4 Chat State Representation
{
  "questions": [
    { "id": "scale_ref", "text": "What is the plate width in mm?" }
  ],
  "answers": [
    { "id": "scale_ref", "value": 100 }
  ]
}

9. Intermediate Language Emission (MML-JSON)
9.1 Responsibilities

Convert pixel-based geometry into real units

Normalize standards (e.g., bolt → clearance diameter)

Record provenance (vision vs user decision)

9.2 Design Principle

MML-JSON must be:

Deterministic

Explicit (no hidden defaults)

Traceable (why each value was chosen)

10. Drawing Generation (DXF)
10.1 Library

ezdxf

10.2 Layers

OUTLINE: visible edges (top/front/right views)

HOLES: circles for holes (top view)

BEND: bend lines (center linetype)

CENTER: center lines for holes/features

HIDDEN: hidden edges in projected views

TEXT: annotations

10.2.1 View layout

Third-angle projection (TOP above FRONT, RIGHT on the right of FRONT)

10.3 Annotation Rules

Always include:

Part name

Material

Plate thickness

Hole standard summary

Use simple text, no dimension arrows in MVP

Example:

PART: Bracket
MAT: A5052  t=2.0
HOLES: 4x M5 clearance
BEND: 90deg R=2.0

11. CLI Behavior Details
mml vision

Input: image file

Output: vision.json

No user interaction

mml interact

Input: image or vision.json

Output: mml.json, report.json

Requires answering mandatory questions

mml draw

Input: mml.json

Output: drawing.dxf

mml pipeline

Executes vision → interact → draw in order

12. Testing Strategy
12.1 Testing Philosophy

All tests must be reproducible

No external image assets

Test images are generated programmatically

12.2 Test Image Generation

Use OpenCV to generate synthetic line drawings:

Rectangles for outlines

Circles for holes

Lines for bends

13. Test Cases
Test Case 1: Simple Plate with 4 Holes

Purpose:

Validate end-to-end pipeline

Steps:

Generate rectangle (200x100 px)

Generate 4 equal circles

Answer scale and bolt standard

Expected:

mml.json generated

drawing.dxf contains outline and 4 holes

Test Case 2: Plate with Bend Line

Purpose:

Validate bend recognition and interaction

Steps:

Rectangle + internal straight line

Answer bend angle and thickness

Expected:

bend entry exists in mml.json

BEND layer exists in DXF

Test Case 3: Ambiguous Hole Sizes

Purpose:

Validate ambiguity resolution via chat

Steps:

Two holes with different radii

User selects unified standard

Expected:

All holes normalized to same diameter

report.json records normalization decision

14. Graduation Research Positioning

This MVP demonstrates:

Integration of image-based interpretation and symbolic modeling

A concrete realization of a mechanical modeling language

A reproducible pipeline from abstract intent to manufacturing artifact

This system is not a replacement for CAD,
but a pre-CAD design intent compiler.

15. Future Work (Out of MVP Scope)

Text-based DSL frontend

3D geometry (STEP)

FEM integration

Multi-part assemblies

Learning-based vision model

LLM-driven design suggestions
