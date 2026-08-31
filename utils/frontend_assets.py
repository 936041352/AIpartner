from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
import os


ASSET_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_FRONTEND_IMAGE_BYTES = 15 * 1024 * 1024
# 背景图上方的白色遮罩透明度；设置为 0 可完全关闭遮罩。
BACKGROUND_WHITE_OVERLAY_OPACITY = 0.18


class FrontendAssetError(ValueError):
    pass


class FrontendAssetTooLargeError(FrontendAssetError):
    pass


@dataclass(frozen=True)
class FrontendAssetSpec:
    filename: str
    use_default: bool


ASSET_SPECS = {
    "user_profile": FrontendAssetSpec("user_profile", True),
    "background_image_gal": FrontendAssetSpec(
        "background_image_gal", True
    ),
}


class FrontendAssetStore:
    """读取和保存角色前端图片，不参与角色对话运行时。"""

    def __init__(self, project_root: Path):
        self.default_directory = project_root.resolve() / "frontend" / "imgs"

    @staticmethod
    def _get_spec(asset_type: str) -> FrontendAssetSpec:
        spec = ASSET_SPECS.get(asset_type)
        if spec is None:
            raise FrontendAssetError(f"不支持的前端资源类型：{asset_type}")
        return spec

    @staticmethod
    def _find_image(directory: Path, filename: str) -> Path | None:
        if not directory.is_dir():
            return None
        candidates = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.stem.casefold() == filename.casefold()
            and path.suffix.casefold() in ASSET_SUFFIXES
        ]
        if not candidates:
            return None
        # 上传的新文件优先；扩展名清理失败时也不会继续读取旧图片。
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def resolve(
        self,
        character_directory: Path,
        asset_type: str,
    ) -> Path | None:
        spec = self._get_spec(asset_type)
        character_asset = self._find_image(
            character_directory / "frontend_imgs",
            spec.filename,
        )
        if character_asset is not None:
            return character_asset
        if spec.use_default:
            return self._find_image(self.default_directory, spec.filename)
        return None

    def save(
        self,
        character_directory: Path,
        asset_type: str,
        content: bytes,
    ) -> Path:
        spec = self._get_spec(asset_type)
        if len(content) > MAX_FRONTEND_IMAGE_BYTES:
            raise FrontendAssetTooLargeError("图片不能超过 15 MB。")

        suffix = self._detect_suffix(content)
        target_directory = character_directory / "frontend_imgs"
        target_directory.mkdir(parents=True, exist_ok=True)
        target_path = target_directory / f"{spec.filename}{suffix}"
        temporary_path = target_directory / f".{uuid4().hex}.tmp"

        try:
            temporary_path.write_bytes(content)
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        for path in target_directory.iterdir():
            if (
                path != target_path
                and path.is_file()
                and path.stem.casefold() == spec.filename.casefold()
                and path.suffix.casefold() in ASSET_SUFFIXES
            ):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # 新文件已经原子替换成功；残留旧扩展名不应使上传失败。
                    pass
        return target_path

    @staticmethod
    def _detect_suffix(content: bytes) -> str:
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if content.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        raise FrontendAssetError("只支持 PNG、JPG 和 JPEG 图片。")
