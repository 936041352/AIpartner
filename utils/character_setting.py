from langchain_core.messages import SystemMessage

from .llm import get_thinking_llm


def read_character_setting(character_dir, character_name) -> str:
    setting_path = character_dir / "character_setting.txt"
    try:
        return setting_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError(
            f"未找到 {character_name} 的角色设定文件。"
        ) from error


def summary_character_setting(
    character_dir,
    character_name,
    character_setting=None,
) -> str:
    """
    读取角色设定总结。

    处理逻辑：
    1. 如果总结文件存在、内容不为空，并且修改时间不早于完整角色设定文件，
       直接返回已有总结。
    2. 如果总结文件不存在、内容为空，或者完整角色设定在总结生成后又被修改，
       则调用大模型重新生成总结。
    3. 大模型调用最多尝试 3 次。
    4. 生成成功后，将总结保存到文件并返回。
    5. 连续 3 次调用失败后，抛出异常。
    """
    character_setting_path = character_dir / "character_setting.txt"
    summary_path = character_dir / "character_setting_summary.txt"

    # 总结文件存在且内容不为空时，检查是否仍然有效
    if summary_path.is_file():
        character_setting_summary = summary_path.read_text(
            encoding="utf-8"
        ).strip()

        if character_setting_summary:
            # 完整角色设定文件不存在时，后续统一交给
            # read_character_setting() 处理
            if character_setting_path.is_file():
                summary_mtime = summary_path.stat().st_mtime_ns
                setting_mtime = character_setting_path.stat().st_mtime_ns

                # 总结文件不早于完整设定文件，说明总结仍然是最新的
                if summary_mtime >= setting_mtime:
                    return character_setting_summary

                print(
                    f"[CharacterChatEngine, {character_name}] "
                    "检测到角色设定已修改，将重新生成设定总结..."
                )

    # 没有可用的角色设定总结，检查完整版角色设定
    if character_setting is None:
        character_setting = read_character_setting(character_dir, character_name)
    if not character_setting:
        raise ValueError(
            f"{character_name} 的完整角色设定为空，无法生成角色设定总结。"
        )

    print(f"[CharacterChatEngine, {character_name}] 正在总结角色设定...")
    llm = get_thinking_llm()
    sys_prompt = f"""
你是一名角色设定摘要助手。请根据下方提供的完整角色设定，生成一份简洁、准确的角色摘要。

这份摘要将传递给其他大模型来总结相关角色的记忆，用于帮助它理解角色的基本身份、判断角色会重视哪些对话信息，以及哪些事件可能对角色产生持续影响。

请重点总结以下三个方面：

【基础设定】
- 角色的姓名、身份、职业、种族、年龄或其他明确的基础信息；
- 角色当前的重要处境、职责或社会关系；
- 只保留理解角色对话和行为所必需的背景，不要复述完整经历或世界观。

【性格与价值观】
- 角色稳定、核心的性格特征；
- 角色重视的原则、价值观、信念、目标和行为底线；
- 角色通常如何判断他人和事件，以及这些特征如何影响其情绪、选择和行为；
- 若角色存在明显的矛盾性格或特殊行为倾向，应准确保留。

【关注与厌恶】
- 角色特别关注、重视、喜欢、依赖或希望保护的人、事物、关系和目标；
- 角色讨厌、排斥、恐惧、警惕、反感或无法接受的人、事物和行为；
- 哪些承诺、关系变化、冲突、伤害、帮助、背叛、成功或失败容易被角色长期记住；
- 哪些对话内容会显著影响角色后续的态度、情绪、信任或行为。

请遵循以下要求：

1. 只能依据原始角色设定，不得补充、推测或虚构原文中没有的信息。
2. 优先保留会影响角色对话、情绪、判断、关系和记忆重要性的设定。
3. 不要进行文学化描写，不要评价或分析角色设定的质量。
4. 如果原始设定存在矛盾、模糊或不确定之处，应保留这种不确定性。
5. 总结应控制在 500 字以内；原始设定较简单时可以更短。
6. 直接输出角色摘要，不要输出分析过程、说明、前言或其他无关内容。
7. 语言要简洁、明确、信息密度高。

请严格按照以下格式输出：

【基础设定】
角色姓名、身份、必要背景和当前重要处境。

【性格与价值观】
核心性格、价值观、信念、目标、原则和底线。

【关注与厌恶】
角色关注、重视、喜欢和希望保护的内容；角色讨厌、排斥、恐惧和警惕的内容；容易对其长期记忆和后续态度产生影响的事件。

以下是完整角色设定：

<character_setting>
{character_setting}
</character_setting>
""".strip()

    max_attempts = 3
    last_error: Exception | None = None

    for _ in range(1, max_attempts + 1):
        try:
            response = llm.invoke(
                [SystemMessage(content=sys_prompt)]
            )

            character_setting_summary = response.content.strip()

            if not character_setting_summary:
                raise ValueError("大模型返回的角色设定总结为空。")

            # 确保保存目录存在
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            summary_path.write_text(
                character_setting_summary,
                encoding="utf-8",
            )
            print(f"[CharacterChatEngine, {character_name}] 角色设定总结如下：\n\n{character_setting_summary}\n\n")

            return character_setting_summary

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"生成 {character_name} 的角色设定总结失败，"
        f"已连续尝试 {max_attempts} 次。"
    ) from last_error
