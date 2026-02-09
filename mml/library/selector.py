"""
ライブラリからのAI部品選択。

OpenAIでユーザー意図を解析し、カタログから適切な部品を選択する。
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI

from .catalog import get_catalog, PartDefinition


@dataclass
class SelectedPart:
    """AIが選択した部品（推定パラメータ付き）。"""

    part_id: str
    definition: PartDefinition
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str
    quantity: int = 1


@dataclass
class SelectionResult:
    """AI部品選択の結果。"""

    parts: List[SelectedPart]
    assembly_notes: str
    raw_response: Dict[str, Any]


class PartSelector:
    """
    AI駆動の部品セレクタ。

    ユーザー意図（文章またはビジョン解析）に基づき、
    カタログから適切な部品を選択しパラメータを推定する。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.catalog = get_catalog()

    def select(
        self,
        intent: Dict[str, Any],
        vision_data: Optional[Dict] = None,
        max_parts: int = 10
    ) -> SelectionResult:
        """
        ユーザー意図に基づいて部品を選択する。

        引数:
            intent: interact.py からの意図辞書
            vision_data: 任意のビジョン解析結果
            max_parts: 選択する最大部品数

        戻り値:
            選択部品とパラメータを含むSelectionResult
        """
        # カタログ文脈付きプロンプトを構築
        catalog_summary = self._build_catalog_prompt()
        user_prompt = self._build_user_prompt(intent, vision_data)

        system_prompt = f"""You are a mechanical engineering assistant that selects appropriate parts from a parts library.

{catalog_summary}

Based on the user's intent, select the appropriate parts and infer reasonable parameters.
Consider:
1. What the user wants to build/achieve
2. Which parts are needed
3. Appropriate parameters for each part
4. Quantities needed

Return JSON with this exact structure:
{{
  "parts": [
    {{
      "part_id": "spur_gear",
      "parameters": {{"module": 1.5, "teeth_count": 24}},
      "quantity": 2,
      "confidence": 0.85,
      "reasoning": "Selected for power transmission"
    }}
  ],
  "assembly_notes": "Description of how parts fit together"
}}

Rules:
- Only select parts that exist in the catalog
- Provide reasonable default parameters if not specified
- Set confidence between 0 and 1
- Keep quantity reasonable (usually 1-4)
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result_data = json.loads(response.choices[0].message.content)
            return self._parse_result(result_data)

        except Exception as e:
            # エラー時は空結果を返す
            return SelectionResult(
                parts=[],
                assembly_notes=f"Selection failed: {e}",
                raw_response={"error": str(e)}
            )

    def _build_catalog_prompt(self) -> str:
        """AI向けのカタログ説明を作成する。"""
        lines = ["## Available Parts Library\n"]

        for category in sorted(self.catalog.categories()):
            lines.append(f"\n### {category.title()}")

            for part in self.catalog.by_category(category):
                name = part.get_name("ja")
                lines.append(f"\n**{part.id}** ({name})")

                if part.ai_context:
                    lines.append(f"  使用場面: {part.ai_context}")

                lines.append("  パラメータ:")
                for param_name, param_def in list(part.parameters.items())[:5]:
                    desc = param_def.get("description", {}).get("ja", param_name)
                    default = param_def.get("default")
                    unit = param_def.get("unit", "")
                    lines.append(f"    - {param_name}: {desc} (default: {default}{unit})")

        return "\n".join(lines)

    def _build_user_prompt(
        self,
        intent: Dict[str, Any],
        vision_data: Optional[Dict]
    ) -> str:
        """意図とビジョンからユーザープロンプトを作成する。"""
        parts = ["## ユーザーの設計意図\n"]

        # 意図の要約
        if intent.get("summary"):
            parts.append(f"目的: {intent['summary']}")

        # 主要意図フィールド
        key_fields = [
            ("function_primary", "主機能"),
            ("mechanism_type", "機構タイプ"),
            ("motion_type", "運動タイプ"),
            ("force_type", "荷重タイプ"),
            ("connections", "接続方法"),
            ("gear_module", "歯車モジュール"),
            ("gear_teeth_count", "歯数"),
        ]

        for field, label in key_fields:
            value = intent.get(field)
            if value:
                parts.append(f"{label}: {value}")

        # ビジョンデータ
        if vision_data:
            if vision_data.get("part_hint"):
                parts.append(f"\n画像から推定: {vision_data['part_hint']}")

            holes = vision_data.get("holes", [])
            if holes:
                parts.append(f"検出された穴: {len(holes)}個")

            bends = vision_data.get("bend_lines", [])
            if bends:
                parts.append(f"検出された曲げ線: {len(bends)}本")

        return "\n".join(parts)

    def _parse_result(self, data: Dict) -> SelectionResult:
        """AI応答をSelectionResultに変換する。"""
        selected_parts = []

        for part_data in data.get("parts", []):
            part_id = part_data.get("part_id")
            if not part_id:
                continue

            definition = self.catalog.get(part_id)
            if definition is None:
                continue

            selected_parts.append(SelectedPart(
                part_id=part_id,
                definition=definition,
                parameters=part_data.get("parameters", {}),
                confidence=float(part_data.get("confidence", 0.5)),
                reasoning=part_data.get("reasoning", ""),
                quantity=int(part_data.get("quantity", 1))
            ))

        return SelectionResult(
            parts=selected_parts,
            assembly_notes=data.get("assembly_notes", ""),
            raw_response=data
        )


def select_parts_for_intent(
    intent: Dict[str, Any],
    api_key: str,
    vision_data: Optional[Dict] = None,
    model: str = "gpt-4o"
) -> SelectionResult:
    """
    部品選択の簡易ラッパー関数。

    引数:
        intent: ユーザー意図の辞書
        api_key: OpenAI API キー
        vision_data: 任意のビジョン結果
        model: 使用するOpenAIモデル

    戻り値:
        選択部品を含むSelectionResult
    """
    selector = PartSelector(api_key=api_key, model=model)
    return selector.select(intent, vision_data)


def heuristic_part_selection(
    intent: Dict[str, Any],
    vision_data: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    AIなしのフォールバック・ヒューリスティック部品選択。

    引数:
        intent: ユーザー意図の辞書
        vision_data: 任意のビジョン結果

    戻り値:
        部品選択辞書のリスト
    """
    catalog = get_catalog()
    parts = []
    # ビジョンのヒントを確認
    if vision_data and vision_data.get("part_hint"):
        hint = vision_data["part_hint"].lower()
        matches = catalog.search_keywords([hint])
        if matches:
            parts.append({
                "part_id": matches[0].id,
                "parameters": {},
                "confidence": float(vision_data.get("part_hint_confidence", 0.6)),
                "reasoning": "Inferred from vision analysis",
                "quantity": 1
            })
    # 機構タイプを確認
    mechanism = (intent.get("mechanism_type") or "").lower()

    if "gear" in mechanism or "歯車" in mechanism:
        gear_params = {}
        if intent.get("gear_module"):
            try:
                gear_params["module"] = float(intent["gear_module"])
            except (ValueError, TypeError):
                pass
        if intent.get("gear_teeth_count"):
            try:
                gear_params["teeth_count"] = int(intent["gear_teeth_count"])
            except (ValueError, TypeError):
                pass

        if not any(p["part_id"] == "spur_gear" for p in parts):
            parts.append({
                "part_id": "spur_gear",
                "parameters": gear_params,
                "confidence": 0.6,
                "reasoning": "Mechanism type indicates gear",
                "quantity": 2
            })

    if "shaft" in mechanism or "軸" in mechanism:
        if not any(p["part_id"] == "shaft" for p in parts):
            parts.append({
                "part_id": "shaft",
                "parameters": {},
                "confidence": 0.5,
                "reasoning": "Mechanism requires shaft",
                "quantity": 1
            })

    if "bearing" in mechanism or "軸受" in mechanism:
        if not any(p["part_id"] == "bearing" for p in parts):
            parts.append({
                "part_id": "bearing",
                "parameters": {},
                "confidence": 0.5,
                "reasoning": "Mechanism requires bearing",
                "quantity": 2
            })
    # 接続方法を確認
    connections = (intent.get("connections") or "").lower()
    if "bolt" in connections or "ボルト" in connections:
        parts.append({
            "part_id": "bolt",
            "parameters": {"size": "M5"},
            "confidence": 0.7,
            "reasoning": "Connection method requires bolts",
            "quantity": 4
        })
        parts.append({
            "part_id": "nut",
            "parameters": {"size": "M5"},
            "confidence": 0.7,
            "reasoning": "Bolts require nuts",
            "quantity": 4
        })

    return parts
