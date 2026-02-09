"""
部品定義を読み込み、索引化するカタログモジュール。

主な責務:
1. library/parts/ から全てのJSON部品定義を読み込む
2. キーワード検索用の索引を構築する
3. ID・カテゴリ・キーワードで検索するAPIを提供する
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PartDefinition:
    """部品定義のインメモリ表現。"""

    id: str
    category: str
    name: Dict[str, str]
    description: Dict[str, str]
    keywords: List[str]
    ai_context: str
    parameters: Dict[str, Any]
    constraints: List[Dict]
    generator: Dict[str, str]
    compatible_with: List[str] = field(default_factory=list)
    assembly_hints: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "PartDefinition":
        """JSON辞書からPartDefinitionを生成する。"""
        return cls(
            id=data["id"],
            category=data["category"],
            name=data.get("name", {}),
            description=data.get("description", {}),
            keywords=data.get("keywords", []),
            ai_context=data.get("ai_context", ""),
            parameters=data.get("parameters", {}),
            constraints=data.get("constraints", []),
            generator=data["generator"],
            compatible_with=data.get("compatible_with", []),
            assembly_hints=data.get("assembly_hints", {}),
        )

    def get_name(self, lang: str = "ja") -> str:
        """ローカライズ名を取得する。"""
        return self.name.get(lang) or self.name.get("en") or self.id

    def get_description(self, lang: str = "ja") -> str:
        """ローカライズ説明を取得する。"""
        return self.description.get(lang) or self.description.get("en") or ""


class PartsCatalog:
    """
    全ての部品定義を扱うカタログの中心クラス。

    使用例:
        catalog = PartsCatalog()
        catalog.load()  # library/parts/ から全件読み込み

        # IDで取得
        gear = catalog.get("spur_gear")

        # キーワードで検索
        matches = catalog.search_keywords(["gear", "rotation"])

        # カテゴリ内の全件取得
        all_gears = catalog.by_category("gears")
    """

    def __init__(self, parts_dir: Optional[str] = None):
        if parts_dir is None:
            parts_dir = Path(__file__).parent / "parts"
        self.parts_dir = Path(parts_dir)
        self._parts: Dict[str, PartDefinition] = {}
        self._keyword_index: Dict[str, List[str]] = {}  # キーワード -> [part_id]
        self._category_index: Dict[str, List[str]] = {}  # カテゴリ -> [part_id]

    def load(self) -> None:
        """JSONファイルから部品定義を読み込む。"""
        self._parts.clear()
        self._keyword_index.clear()
        self._category_index.clear()

        if not self.parts_dir.exists():
            return

        for category_dir in self.parts_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue
            for json_file in category_dir.glob("*.json"):
                self._load_part(json_file)

        self._build_indices()

    def _load_part(self, path: Path) -> None:
        """単一の部品定義を読み込む。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            part = PartDefinition.from_json(data)
            self._parts[part.id] = part
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load part from {path}: {e}")

    def _build_indices(self) -> None:
        """キーワードとカテゴリの索引を構築する。"""
        for part_id, part in self._parts.items():
            # キーワード索引
            for kw in part.keywords:
                kw_lower = kw.lower()
                if kw_lower not in self._keyword_index:
                    self._keyword_index[kw_lower] = []
                self._keyword_index[kw_lower].append(part_id)

            # カテゴリ索引
            if part.category not in self._category_index:
                self._category_index[part.category] = []
            self._category_index[part.category].append(part_id)

    def get(self, part_id: str) -> Optional[PartDefinition]:
        """IDで部品を取得する。"""
        return self._parts.get(part_id)

    def search_keywords(self, keywords: List[str]) -> List[PartDefinition]:
        """指定キーワードに一致する部品を検索する。"""
        matching_ids = set()
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in self._keyword_index:
                matching_ids.update(self._keyword_index[kw_lower])
        return [self._parts[pid] for pid in matching_ids]

    def by_category(self, category: str) -> List[PartDefinition]:
        """カテゴリ内の全ての部品を取得する。"""
        part_ids = self._category_index.get(category, [])
        return [self._parts[pid] for pid in part_ids]

    def all_parts(self) -> List[PartDefinition]:
        """読み込み済みの部品を全て取得する。"""
        return list(self._parts.values())

    def categories(self) -> List[str]:
        """全てのカテゴリ名を取得する。"""
        return list(self._category_index.keys())

    def get_catalog_summary(self, lang: str = "ja") -> str:
        """AIコンテキスト用の概要文字列を生成する。"""
        lines = ["Available Parts Library:\n"]
        for category in sorted(self._category_index.keys()):
            lines.append(f"\n## {category.title()}")
            for part_id in self._category_index[category]:
                part = self._parts[part_id]
                name = part.get_name(lang)
                lines.append(f"- {part_id}: {name}")
                if part.ai_context:
                    context = part.ai_context[:100]
                    if len(part.ai_context) > 100:
                        context += "..."
                    lines.append(f"  Context: {context}")
        return "\n".join(lines)

    def get_parts_for_ai(self) -> List[Dict[str, Any]]:
        """AIプロンプト用に整形した部品情報を取得する。"""
        result = []
        for part in self._parts.values():
            params_desc = []
            for param_name, param_def in part.parameters.items():
                desc = param_def.get("description", {}).get("ja", param_name)
                default = param_def.get("default")
                unit = param_def.get("unit", "")
                params_desc.append(f"{param_name}: {desc} (default: {default}{unit})")

            result.append(
                {
                    "id": part.id,
                    "category": part.category,
                    "name": part.get_name("ja"),
                    "ai_context": part.ai_context,
                    "parameters": params_desc,
                    "compatible_with": part.compatible_with,
                }
            )
        return result


# グローバルシングルトン
_catalog: Optional[PartsCatalog] = None


def get_catalog() -> PartsCatalog:
    """グローバルなカタログインスタンスを取得または作成する。"""
    global _catalog
    if _catalog is None:
        _catalog = PartsCatalog()
        _catalog.load()
    return _catalog


def reload_catalog() -> PartsCatalog:
    """カタログを強制的に再読み込みする。"""
    global _catalog
    _catalog = PartsCatalog()
    _catalog.load()
    return _catalog
