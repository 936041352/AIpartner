import json
import math
import os
from pathlib import Path
from uuid import uuid4


DEFAULT_PORTRAIT_LAYOUT = {
    "offset_x_vw": 0.0,
    "offset_y_vh": 0.0,
    "scale": 1.0,
}
PORTRAIT_LAYOUT_LIMITS = {
    "offset_x_vw": (-45.0, 45.0),
    "offset_y_vh": (-35.0, 35.0),
    "scale": (0.5, 2.0),
}


class PortraitLayoutError(ValueError):
    pass


class PortraitLayoutStore:
    """读取和原子保存角色共用的前端立绘布局。"""

    filename = "portrait_layout.json"

    @staticmethod
    def _normalize(layout: dict) -> dict[str, float]:
        if not isinstance(layout, dict):
            raise PortraitLayoutError("立绘布局格式无效。")

        normalized = {}
        for field, (minimum, maximum) in PORTRAIT_LAYOUT_LIMITS.items():
            try:
                value = float(layout[field])
            except (KeyError, TypeError, ValueError) as error:
                raise PortraitLayoutError(
                    f"立绘布局参数 {field} 无效。"
                ) from error
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise PortraitLayoutError(
                    f"立绘布局参数 {field} 超出允许范围。"
                )
            normalized[field] = value
        return normalized

    def load(self, character_directory: Path) -> dict[str, float]:
        path = character_directory / "frontend_imgs" / self.filename
        if not path.is_file():
            return dict(DEFAULT_PORTRAIT_LAYOUT)
        try:
            return self._normalize(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, PortraitLayoutError):
            # 展示配置损坏时回退默认值，不能阻断角色对话页面加载。
            return dict(DEFAULT_PORTRAIT_LAYOUT)

    def save(
        self,
        character_directory: Path,
        layout: dict,
    ) -> dict[str, float]:
        normalized = self._normalize(layout)
        target_directory = character_directory / "frontend_imgs"
        target_directory.mkdir(parents=True, exist_ok=True)
        target_path = target_directory / self.filename
        temporary_path = target_directory / f".{uuid4().hex}.tmp"

        try:
            temporary_path.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return normalized
