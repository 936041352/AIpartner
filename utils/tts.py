import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from faster_qwen3_tts import FasterQwen3TTS


class Qwen3TTSGenerator:
    # 一般来说一秒可以说4个汉字
    CHARACTERS_PER_SECOND = 4.0
    # FasterQwen3TTS 的音频编码速率约为每秒 12 token
    CODEC_TOKENS_PER_SECOND = 12

    def __init__(
        self,
        model_dir: str | Path,
        language: str = "Chinese",
        min_audio_duration: float = 1.5,
        max_audio_duration: float = 20.0,
        duration_multiplier: float = 1.5,
    ):
        """
        初始化共享 Qwen3-TTS 模型。
        模型以及 CUDA Graph 在整个程序生命周期中只初始化一次。
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，Qwen3-TTS 无法启动。")

        self.model_dir = Path(model_dir)
        self.language = language
        self.min_audio_duration = min_audio_duration  # 语音合成时间最大值
        self.max_audio_duration = max_audio_duration  # 语音合成时间最小值
        self.duration_multiplier = duration_multiplier

        if min_audio_duration <= 0 or max_audio_duration < min_audio_duration:
            raise ValueError("语音时长范围无效。")
        if duration_multiplier <= 0:
            raise ValueError("语音时长倍率必须大于 0。")

        # {
        #     "char_name": {
        #         "default": (voice_prompt, ref_text),
        #         "happy":   (voice_prompt, ref_text),
        #     }
        # }
        self.prompt_registry = {}

        print(f"[语音模型] 使用 GPU：{torch.cuda.get_device_name(0)}")
        print("[语音模型] 正在加载 Qwen3-TTS...")

        self.tts = FasterQwen3TTS.from_pretrained(
            str(self.model_dir),
            device="cuda",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )

        print("[语音模型] 正在预热 CUDA Graph...")
        self.tts.warmup(prefill_len=100)

        print("[语音模型] Qwen3-TTS 加载完成。")

    def generate(
        self,
        ref_audio: str,
        ref_text: str,
        gen_text: str,
        file_wave: str = "output.wav",
        speed: float = 1.0,
        seed: int | None = None,
        **kwargs,
    ):
        """
        使用已缓存的 Voice Prompt 生成语音。

        ref_audio / ref_text 参数继续保留，以兼容原有 F5-TTS 调用方式。
        实际 Voice Prompt 和参考文本由 ref_audio 对应的缓存确定。
        """

        # ref_audio:
        # characters/char_name/ref_audios/happy.wav
        #
        # character_name = char_name
        # audio_name = happy
        audio_path = Path(ref_audio)
        character_name = audio_path.parent.parent.name
        audio_name = audio_path.stem

        character_prompts = self.prompt_registry.get(character_name)
        if character_prompts is None:
            raise ValueError(
                f"角色 '{character_name}' 尚未初始化语音资源。"
            )

        prompt_data = character_prompts.get(audio_name)
        if prompt_data is None:
            raise ValueError(
                f"角色 '{character_name}' 不存在已缓存的参考语音："
                f"{audio_name}"
            )

        voice_prompt, cached_ref_text = prompt_data

        # 保留旧F5-TTS接口参数，但 FasterQwen 当前没有 speed 和 seed 参数。
        if speed != 1.0:
            print("[语音提示] Qwen3-TTS 当前忽略 speed 参数。")
        if seed is not None:
            print("[语音提示] Qwen3-TTS 当前忽略 seed 参数。")

        language = kwargs.pop("language", self.language)

        audio, sample_rate = self._generate_with_duration_limit(
            gen_text=gen_text,
            language=language,
            cached_ref_text=cached_ref_text,
            voice_prompt=voice_prompt,
            generation_kwargs=kwargs,
        )

        output_path = Path(file_wave)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            output_path,
            audio,
            sample_rate,
        )

        # 保持和原先文档F5-TTS中的返回形式相近。
        # Qwen3-TTS 没有 F5-TTS 的 spec。
        return audio, sample_rate, None

    def _generate_with_duration_limit(
        self,
        gen_text: str,
        language: str,
        cached_ref_text: str,
        voice_prompt,
        generation_kwargs: dict,
    ) -> tuple[np.ndarray, int]:
        """按文本长度限制语音时长；异常时最多重新生成一次。"""
        text_length = max(
            1,
            sum(not character.isspace() for character in gen_text),
        )
        estimated_duration = (
            text_length / self.CHARACTERS_PER_SECOND
        )
        duration_limit = min(
            self.max_audio_duration,
            max(
                self.min_audio_duration,
                estimated_duration * self.duration_multiplier,
            ),
        )

        # Qwen3-TTS 每秒约生成 12 个 codec token，从生成阶段限制异常延长。
        kwargs = dict(generation_kwargs)
        dynamic_max_tokens = math.ceil(
            duration_limit * self.CODEC_TOKENS_PER_SECOND
        )
        configured_max_tokens = kwargs.get("max_new_tokens")
        if configured_max_tokens is not None:
            dynamic_max_tokens = min(
                dynamic_max_tokens,
                configured_max_tokens,
            )
        kwargs["max_new_tokens"] = dynamic_max_tokens

        # 去除文本里常见的表情
        gen_text = remove_emoji(gen_text)

        for attempt in range(2):
            audio_list, sample_rate = self.tts.generate_voice_clone(
                text=gen_text,
                language=language,
                ref_text=cached_ref_text,
                voice_clone_prompt=voice_prompt,
                **kwargs,
            )
            audio = np.asarray(audio_list[0]).squeeze()
            actual_duration = len(audio) / sample_rate

            if actual_duration <= duration_limit:
                return audio, sample_rate

            if attempt == 0:
                print(
                    f"[语音生成警告] 语音时长 {actual_duration:.2f} 秒超过 "
                    f"{duration_limit:.2f} 秒，正在重新生成。"
                )

        print(
            "[语音生成警告] 连续两次语音合成异常，"
            f"异常文本为：{gen_text}"
        )
        return audio, sample_rate

    def init_character(self, character_dir: str | Path) -> bool:
        """
        初始化一个角色的全部参考语音，并缓存对应 Voice Prompt。

        character_dir:
            characters/char_name
        """
        character_dir = Path(character_dir)
        character_name = character_dir.name
        if character_name in self.prompt_registry.keys():
            print(f'[语音生成] {character_name} 的 Voice Prompt 资源已存在，先清理再重新构建')
            self.delete_character(character_name)

        ref_voices = self._load_ref_voices(character_dir)

        if not ref_voices:
            print(
                f"[语音资源错误] 角色 '{character_name}' "
                "没有有效的参考语音。"
            )
            return False

        character_prompts = {}

        print(
            f"[语音资源] 正在初始化角色 "
            f"'{character_name}' 的参考语音..."
        )

        for audio_name, (audio_path, ref_text) in ref_voices.items():
            try:
                voice_prompt = self._create_voice_prompt(
                    audio_path,
                    ref_text,
                )

                character_prompts[audio_name] = (
                    voice_prompt,
                    ref_text,
                )

                print(
                    f"[语音资源] 已缓存："
                    f"{character_name}/{audio_name}"
                )

            except Exception as e:
                print(
                    f"[语音资源警告] "
                    f"{audio_path.name} 初始化失败：{e}"
                )

        if not character_prompts:
            print(
                f"[语音资源错误] 角色 '{character_name}' "
                "没有成功初始化任何参考语音。"
            )
            return False

        self.prompt_registry[character_name] = character_prompts

        print(
            f"[语音资源] 角色 '{character_name}' 初始化完成，"
            f"共缓存 {len(character_prompts)} 个 Voice Prompt。"
        )

        return True

    def delete_character(self, character_name: str):
        """
        删除指定角色的 Voice Prompt 缓存。
        """
        self.prompt_registry.pop(character_name, None)
        print(f'[清除角色资源] {character_name} 的 Voice Prompt 资源已清理')

    def _load_ref_voices(
        self,
        character_dir: Path,
    ) -> dict:
        """
        扫描：
            character_dir/ref_audios

        返回：
            {
                "default": (audio_path, ref_text),
                "happy": (audio_path, ref_text),
            }
        """
        voice_dir = character_dir / "ref_audios"

        if not voice_dir.is_dir():
            print(
                f"[语音资源错误] 参考语音目录不存在：{voice_dir}"
            )
            return {}

        # 同名 TXT：
        # happy.wav -> happy.txt
        transcript_paths = {
            path.stem.casefold(): path
            for path in voice_dir.iterdir()
            if path.is_file()
            and path.suffix.casefold() == ".txt"
            and not path.stem.casefold().startswith("description")
        }

        # default 优先，其余按文件名排序。
        audio_paths = sorted(
            (
                path
                for path in voice_dir.iterdir()
                if path.is_file()
                and path.suffix.casefold() in {".wav", ".mp3"}
                and not path.stem.casefold().startswith("description")
            ),
            key=lambda path: (
                path.stem.casefold() != "default",
                path.name.casefold(),
            ),
        )

        voices = {}

        for audio_path in audio_paths:
            transcript_path = transcript_paths.get(
                audio_path.stem.casefold()
            )

            if transcript_path is None:
                print(
                    f"[语音资源警告] "
                    f"{audio_path.name} 缺少同名 TXT，已忽略。"
                )
                continue

            ref_text = transcript_path.read_text(
                encoding="utf-8-sig"
            ).strip()

            if not ref_text:
                print(
                    f"[语音资源警告] "
                    f"{transcript_path.name} 内容为空，已忽略。"
                )
                continue

            voices[audio_path.stem] = (
                audio_path,
                ref_text,
            )

        return voices

    def _create_voice_prompt(
        self,
        audio_path: Path,
        ref_text: str,
    ):
        """
        创建 Full ICL Voice Prompt。

        FasterQwen 的 ICL 路径默认会给参考音频尾部追加
        约 0.5 秒静音；这里预计算 Prompt 时保持相同行为。
        """
        audio, sr = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        # 转单声道。
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        silence = np.zeros(
            int(sr * 0.5),
            dtype=np.float32,
        )

        audio = np.concatenate(
            [audio, silence]
        )

        return self.tts.model.create_voice_clone_prompt(
            ref_audio=(audio, sr),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )


# 覆盖聊天文本中绝大多数常见 Emoji / 表情
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # 符号和象形文字
    "\U0001F600-\U0001F64F"  # 表情脸
    "\U0001F680-\U0001F6FF"  # 交通和地图
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"  # 补充符号，如 🤣、🤔 等
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"  # 杂项符号，如 ☀
    "\U00002700-\U000027BF"  # 装饰符号，如 ✨
    "\U0001F1E6-\U0001F1FF"  # 国旗区域指示符
    "]+",
    flags=re.UNICODE,
)

# Emoji 中可能附带的特殊字符
EMOJI_EXTRA_PATTERN = re.compile(
    "["
    "\uFE0F"                 # Variation Selector-16
    "\u200D"                 # Zero Width Joiner
    "\U0001F3FB-\U0001F3FF"  # 肤色修饰符
    "]",
    flags=re.UNICODE,
)


def remove_emoji(text: str) -> str:
    """
    删除文本中的常见 Emoji。

    Args:
        text: 原始文本

    Returns:
        去除 Emoji 后的文本
    """
    text = EMOJI_PATTERN.sub("", text)
    text = EMOJI_EXTRA_PATTERN.sub("", text)
    return text
