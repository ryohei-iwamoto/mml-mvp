# MML-MVP: 機械モデリング言語（卒業研究MVP）

## 0. コンセプト

機械設計では、最終成果物として「寸法が確定した図面/CAD」が作られる。
しかし設計の初期段階では、実際に重要なのは以下である。

- どんな動き・機能を実現したいのか（意図）
- そのためにどんな機構を選ぶのか（構造）
- どんな制約を満たす必要があるのか（加工・規格・強度・組立）
- 材料や製法が変わっても成立する抽象性（柔軟性）

図面に落とし込むと寸法が固定されるため、材料や製法の変更に追従しづらい。
そこで本研究では、ラフスケッチと対話入力から「設計意図」を抽象言語として保持し、
その抽象モデルから図面・3Dモデルを生成できるパイプラインを作る。

本システムはCADの代替ではなく、**CAD前段階の設計意図コンパイラ**である。

**ラフ画像 → 認識 → 対話で曖昧さ解消 → 中間言語(MML-JSON) → 図面(DXF) / 画像(PNG) / 3Dモデル(STL) 出力**

---

## 1. 本MVPにおける「機械モデリング言語」とは

このMVPにおける「機械モデリング言語」は、CADのように形状操作を列挙する言語ではない。
設計の意図を、次のレイヤで統合表現するための中間言語である。

| レイヤ | 説明 | 例 |
|--------|------|-----|
| Feature | 何を作るか | plate, bracket, gear, RobotArm |
| Interface | 何と繋ぐか | bolt pattern, hole standard |
| Constraint | 何を満たすべきか | 板厚 >= 2mm, 曲げR >= t, 端距離 >= 2d |
| Intent | 設計意図（91フィールド） | 機構種別, 運動タイプ, 荷重条件, サブコンポーネント |
| Provenance | 各値の出所 | 画像推定 or 対話確定（追跡可能性） |

MVPでは中間言語として **MML-JSON** を採用する。

---

## 2. 実装済み機能

### 入力
- **画像アップロード**: ラフスケッチ画像（PNG, JPG, BMP）からの特徴認識
- **AI画像認識**: OpenAI GPT-4o-mini による部品種別の推定（写真にも対応）
- **対話チャット**: AIによる動的質問生成、またはルールベースの質問
- **MML-JSONアップロード**: 既存のMML-JSONを読み込んでの修正・図面生成
- **直接フォーム入力**: パラメータを直接指定して図面生成

### 出力フォーマット
- **MML-JSON** (`mml.json`): 設計意図を含む中間言語データ
- **DXF** (`drawing.dxf`): 第三角法による三面図（上面図・正面図・右側面図）、レイヤ分け済み
- **PNG** (`drawing.png`): DXF内容のラスタ画像
- **STL** (`model.stl`): 3Dメッシュデータ（板厚押出し、穴ボス、歯車歯形など対応）
- **レポート** (`report.json`): 推論過程・Q&Aログ・確信度の記録

### 対応部品タイプ
- **プレート / ブラケット**: 板金部品（穴・曲げ対応）
- **歯車（Gear）**: 歯形プロファイル付き3D生成
- **ロボットアーム（RobotArm）**: 12部品のマルチコンポーネント組立
  - base, joint, link, end_effector, actuator, gear, shaft, bearing
  - `assembly.stl` として一体組立ファイルを生成

### 画像認識（Vision）
- **外形検出**: 最大輪郭抽出、Douglas-Peucker近似
- **穴検出**: 3段階（輪郭ベース → 階層ベース → ハフ円）、重複排除、最大50穴
- **曲げ線検出**: 確率的ハフ直線検出、外形内フィルタ
- **AI認識**: GPT-4o-mini によるパーツ種別ヒント

### 設計意図キャプチャ（91フィールド）
意味的意図、運動学、力学、構造、組立、荷重ケース、品質、ライフサイクル、熱/電気、
ユーザー体験、歯車固有パラメータなど、91の設計意図フィールドを対話で収集する。

### AI統合（OpenAI GPT-4o-mini）
- 画像からの部品推定
- 動的な質問生成（設計意図 / 幾何形状）
- ユーザー入力の分類（回答 / 質問 / 明確化要求）
- サブコンポーネント提案
- ロボットアーム構成提案・寸法最適化

---

## 3. 全体パイプライン

```
入力（画像 / JSON）
  → Vision: 外形・穴・曲げ線の認識 → vision.json
  → Interact: 対話による設計意図の収集（91フィールド）
  → Emit: MML-JSON + report.json の生成
  → Draw: DXF / PNG / STL の出力
```

---

## 4. システム構成

### Webアプリ（Flask）

ブラウザから操作できるWebインターフェースを提供する。

#### ワークフロー

| ワークフロー | 説明 |
|-------------|------|
| **Model** (`/model`) | 画像アップロード → AI/ルールベース対話 → MML-JSON生成 → 図面出力 |
| **Draw** (`/draw`) | MML-JSONを読み込み → 対話で幾何形状を調整 → DXF/PNG/STL出力 |

#### 主要エンドポイント

| メソッド | パス | 機能 |
|---------|------|------|
| GET | `/` | ホーム画面 |
| GET | `/model` | Modelワークフロー開始画面 |
| POST | `/model/vision` | 画像認識 + 対話開始 |
| POST | `/model/emit` | MML-JSON生成 + 図面出力 |
| GET | `/draw` | Drawワークフロー開始画面 |
| POST | `/draw/load` | MML-JSON読み込み + 対話開始 |
| POST | `/draw/chat` | 対話による幾何形状の調整 |
| POST | `/draw/generate` | DXF/PNG/STL生成 |
| POST | `/draw/run` | 直接パラメータ入力での生成 |
| GET | `/outputs/<run_id>/<filename>` | 生成ファイルのダウンロード |

### CLI

```bash
mml vision input.png -o out/         # → vision.json
mml interact input.png --chat=rule -o out/  # → mml.json, report.json
mml draw out/mml.json -o out/        # → drawing.dxf
mml pipeline input.png --chat=rule -o out/  # → 全パイプライン実行
mml library list                     # → 部品ライブラリ一覧
```

---

## 5. MML-JSON スキーマ

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
  "intent": {
    "intent_summary": "...",
    "function_primary": "...",
    "mechanism_type": "...",
    "subcomponents": ["base", "joint", "link", "..."],
    "arm_config": { "dof": 5, "drive": "servo", "reach_mm": 300 }
  },
  "provenance": {
    "vision": { "file": "input.png", "version": "0.1" },
    "chat": { "mode": "ai", "questions": [], "answers": [] }
  }
}
```

---

## 6. 出力フォーマット詳細

### DXF (`drawing.dxf`)
第三角法による三面図を出力する。

| レイヤ | 内容 |
|--------|------|
| OUTLINE | 外形線 |
| HOLES | 穴の円 |
| BEND | 曲げ線（一点鎖線） |
| CENTER | 穴・フィーチャの中心線 |
| HIDDEN | 投影図の隠れ線 |
| TEXT | 部品名・材料・穴情報・曲げ情報の注記 |
| VIEW_FRAME | ビュー枠 |

### STL (`model.stl`)
3Dメッシュを出力する。部品タイプごとに適切な形状を生成する。

| 部品タイプ | 3D形状 |
|-----------|--------|
| プレート / ブラケット | 板厚押出し + 穴 |
| リンク（link） | 角丸矩形 + ボスリング |
| ジョイント（joint） | 円柱 + 中心穴 + カラー |
| ベース（base） | 矩形 + 4穴 + スタンドオフ |
| 歯車（gear） | 歯形プロファイル押出し |
| シャフト（shaft） | 円柱 |
| ベアリング（bearing） | リング + 内径穴 |
| エンドエフェクタ | 小型ボックス |
| アクチュエータ | 円柱 |

ロボットアームの場合、全コンポーネントを結合した `assembly.stl` も生成する。

---

## 7. ディレクトリ構成

```
product/
├── app.py                  # Flaskメインアプリケーション
├── requirements.txt        # Python依存パッケージ
├── .env.example            # 環境変数テンプレート
├── mml/
│   ├── vision.py           # 画像認識（OpenCV）
│   ├── ai_vision.py        # AI画像認識（GPT-4o-mini）
│   ├── interact.py         # 対話モジュール（91意図フィールド）
│   ├── intent.py           # 部品推定・ライブラリマッピング
│   ├── emit.py             # MML-JSON生成
│   ├── draw.py             # DXF/PNG図面生成
│   ├── stl.py              # STL 3Dメッシュ生成
│   ├── pipeline.py         # パイプラインオーケストレーション
│   ├── cli.py              # CLIインターフェース
│   ├── utils.py            # ユーティリティ関数
│   └── library/            # 部品ライブラリ
│       ├── catalog.py      # 部品カタログ
│       ├── selector.py     # AI/ヒューリスティック部品選択
│       ├── validators.py   # パラメータバリデーション
│       ├── generator.py    # メッシュ生成
│       ├── generators/     # 部品タイプ別ジェネレータ
│       └── parts/          # 部品定義
├── templates/
│   ├── index.html          # ホーム画面
│   ├── model_start.html    # Model開始画面
│   ├── model_chat.html     # Model対話画面
│   ├── model_result.html   # Model結果画面
│   ├── draw_start.html     # Draw開始画面
│   ├── draw_chat.html      # Draw対話画面
│   ├── draw_result.html    # Draw結果画面
│   └── result.html         # レガシー結果画面
├── static/
│   └── style.css           # スタイルシート
├── outputs/                # 生成物出力先
└── tests/                  # テスト
```

---

## 8. セットアップ

### 必要環境
- Python 3.10+
- OpenAI APIキー（AI機能利用時）

### インストール

```bash
cd product
pip install -r requirements.txt
```

### 環境変数設定

```bash
cp .env.example .env
# .env を編集して OPENAI_API_KEY を設定
```

### 起動

```bash
# Webアプリ
python app.py
# → http://localhost:5000

# CLI
python -m mml pipeline input.png --chat=rule -o out/
```

### 主な依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| flask | Webフレームワーク |
| opencv-python-headless | 画像認識 |
| numpy | 数値計算 |
| ezdxf | DXFファイル生成 |
| trimesh | 3Dメッシュ操作 |
| shapely | ポリゴンジオメトリ |
| mapbox-earcut | ポリゴン三角形分割 |
| openai | OpenAI API クライアント |

---

## 9. 出力ディレクトリ構造

```
outputs/<run_id>/
├── input.png              # 入力画像
├── vision.json            # 画像認識結果
├── mml.json               # MML-JSON（中間言語）
├── report.json            # 推論レポート
├── drawing.dxf            # 三面図（DXF）
├── drawing.png            # 三面図（PNG）
├── model.stl              # 3Dモデル（STL）
├── <component>_0/         # マルチコンポーネント時の各部品
│   ├── mml.json
│   ├── drawing.dxf
│   ├── drawing.png
│   ├── model.stl
│   └── drawing_report.json
└── assembly.stl           # 組立モデル（ロボットアーム時）
```

---

## 10. テスト

テスト画像はプログラムで生成するため、外部画像アセット不要。

```bash
cd product
python -m pytest tests/
```

### テストケース
1. **4穴付きシンプルプレート**: パイプライン一気通貫の検証
2. **曲げ線付きプレート**: 曲げ認識と対話の検証
3. **曖昧な穴サイズ**: 対話による曖昧さ解消の検証

---

## 11. 卒業研究における位置づけ

本MVPが示すもの:
- 画像ベースの解釈とシンボリックモデリングの統合
- 機械モデリング言語の具体的な実現
- 抽象的な設計意図から製造成果物（図面・3Dモデル）への再現可能なパイプライン
- AI対話による設計意図の段階的詳細化
- マルチコンポーネント組立（ロボットアーム）の自動生成
