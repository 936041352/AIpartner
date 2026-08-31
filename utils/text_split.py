import re


# 一级边界：通常表示完整语义或明显停顿
_STRONG_PUNCT = set("。！？!?…～")

# 二级边界：句内停顿
_WEAK_PUNCT = set("，,、；;：:—")

# 句末标点之后应继续附着在前一片段的符号
_CLOSING_MARKS = set('”’"\'」』）》】)}〕〉')

# 中文无标点长句的启发式候选边界
_SOFT_BREAK_WORDS = (
    "但是",
    "不过",
    "因此",
    "所以",
    "然后",
    "同时",
    "此外",
    "而且",
    "以及",
    "并且",
    "如果",
    "那么",
    "虽然",
    "由于",
    "因为",
)

# 用单字节控制字符临时保护特殊结构中的标点。
# 文本预处理阶段会先删除原始控制字符，因此不会发生占位符冲突。
_PROTECT_MAP = {
    ".": "\x01",
    ",": "\x02",
    ":": "\x03",
    ";": "\x04",
    "?": "\x05",
    "!": "\x06",
}

_PROTECT_TRANS = str.maketrans(_PROTECT_MAP)
_RESTORE_TRANS = str.maketrans(
    {protected: original for original, protected in _PROTECT_MAP.items()}
)


# 需要保护内部标点的常见结构
_PROTECTED_PATTERN = re.compile(
    r"""
    (?P<url>
        (?:https?://|www\.)
        [^\s<>"'，。！？；：]+
    )
    |
    (?P<email>
        [A-Za-z0-9._%+-]+
        @
        [A-Za-z0-9.-]+
        \.
        [A-Za-z]{2,}
    )
    |
    (?P<number>
        (?<![\w])
        [-+]?
        (?:
            \d{1,3}(?:,\d{3})+
            |
            \d+
        )
        (?:\.\d+)?
        (?:%|[A-Za-z]+)?
    )
    |
    (?P<title>
        \b
        (?:
            Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|
            No|Fig|Eq|Dept|Inc|Ltd|Co
        )
        \.
    )
    |
    (?P<abbr>
        \b(?:e\.g|i\.e|a\.m|p\.m|etc|vs)\.
    )
    |
    (?P<initialism>
        \b(?:[A-Za-z]\.){2,}
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


_SOFT_WORD_PATTERN = "|".join(
    re.escape(word)
    for word in sorted(_SOFT_BREAK_WORDS, key=len, reverse=True)
)

# 无标点兜底阶段的原子项。
# URL、邮箱、数字、英文缩写和普通英文单词尽量不拆开。
_ATOM_PATTERN = re.compile(
    rf"""
    (?:https?://|www\.)[^\s<>"'，。！？；：]+
    |
    [A-Za-z0-9._%+-]+
    @
    [A-Za-z0-9.-]+
    \.
    [A-Za-z]{{2,}}
    |
    (?:[A-Za-z]\.){{2,}}
    |
    (?:
        Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|
        No|Fig|Eq|Dept|Inc|Ltd|Co|etc|vs
    )
    \.
    |
    (?:e\.g|i\.e|a\.m|p\.m)\.
    |
    [-+]?
    (?:
        \d{{1,3}}(?:,\d{{3}})+
        |
        \d+
    )
    (?:\.\d+)?
    (?:%|[A-Za-z]+)?
    |
    {_SOFT_WORD_PATTERN}
    |
    [A-Za-z0-9]+
    (?:[’'][A-Za-z0-9]+)*
    (?:-[A-Za-z0-9]+)*
    |
    [ \t]+
    |
    .
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


_TRAILING_PUNCT_PATTERN = re.compile(
    r'([。！？!?…，,、；;：:—]+[”’"\'」』）》】)}〕〉]*)$'
)


def _utf8_len(text: str) -> int:
    """返回文本的 UTF-8 字节数。"""
    return len(text.encode("utf-8"))


def _remove_balanced_brackets(text: str) -> str:
    """
    删除成对括号及其内部内容。

    采用逐层删除方式，因此支持嵌套括号；
    未闭合的括号保持原样，避免误删后续全部文本。
    """
    patterns = (
        re.compile(r"\([^()]*\)"),
        re.compile(r"（[^（）]*）"),
        re.compile(r"\[[^\[\]]*\]"),
        re.compile(r"【[^【】]*】"),
        re.compile(r"\{[^{}]*\}"),
    )

    while True:
        old_text = text

        for pattern in patterns:
            text = pattern.sub("", text)

        if text == old_text:
            return text

# 匹配每一行开头的 Markdown 标题、引用和列表标记。
_MARKDOWN_LINE_PREFIX_PATTERN = re.compile(
    r"""
    ^\s{0,3}
    (?:
        \#{1,6}[ \t]+     # Markdown 标题，例如 ## 标题
        |
        >[ \t]*           # Markdown 引用，例如 > 引用
        |
        [-+*][ \t]+       # 无序列表，例如 - 项目
        |
        \d+[.)][ \t]+     # 有序列表，例如 1. 项目或 1) 项目
    )
    """,
    flags=re.MULTILINE | re.VERBOSE,
)


def _normalize_text(
    text: str,
    remove_brackets: bool,
) -> str:
    """清理不适合朗读的格式，同时尽量保留正文语义。"""

    # 统一换行符。
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 删除不可见控制字符，但保留换行和制表符。
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text,
    )

    # Markdown 图片只保留替代文字。
    # 例如：![图片说明](image.png) -> 图片说明
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Markdown 链接只保留可读文字。
    # 例如：[OpenAI](https://openai.com) -> OpenAI
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    if remove_brackets:
        text = _remove_balanced_brackets(text)

    # 删除 HTML 标签，但保留普通比较符号，如 x < 5。
    text = re.sub(
        r"</?[A-Za-z][^>\n]*>",
        " ",
        text,
    )

    # 删除行首 Markdown 标题、引用和列表标记。
    text = _MARKDOWN_LINE_PREFIX_PATTERN.sub("", text)

    # 删除代码块和行内代码标记，但保留其中的文字。
    text = text.replace("```", "").replace("`", "")

    # 删除 Markdown 强调标记，同时避免破坏 snake_case。
    text = re.sub(
        r"(?<!\w)[*_~]{1,3}|[*_~]{1,3}(?!\w)",
        "",
        text,
    )

    # 合并横向空白，但保留换行作为硬语义边界。
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)

    return text.strip()
def _protect_internal_punctuation(text: str) -> str:
    """保护 URL、邮箱、数字和常见英文缩写内部的标点。"""

    def replace(match: re.Match) -> str:
        value = match.group(0)
        suffix = ""

        # URL 正则可能连同正文句末标点一起匹配。
        # 将尾部句末标点重新释放，使其可以参与切分。
        if match.lastgroup == "url":
            while value and value[-1] in ".,;:!?":
                suffix = value[-1] + suffix
                value = value[:-1]

        return value.translate(_PROTECT_TRANS) + suffix

    return _PROTECTED_PATTERN.sub(replace, text)


def _restore_internal_punctuation(text: str) -> str:
    """恢复之前保护的标点。"""
    return text.translate(_RESTORE_TRANS)


def _split_by_punctuation(
    text: str,
    puncts: set[str],
) -> list[str]:
    """
    按指定标点切分。

    连续标点、右引号和右括号会附着在前一个片段，
    避免产生以右引号开头的异常片段。
    """
    result: list[str] = []
    buffer: list[str] = []
    index = 0

    while index < len(text):
        char = text[index]
        buffer.append(char)

        if char in puncts:
            # 收集连续标点，例如 “……” 或 “?!”
            while (
                index + 1 < len(text)
                and text[index + 1] in puncts
            ):
                index += 1
                buffer.append(text[index])

            # 收集句末右引号、右括号
            while (
                index + 1 < len(text)
                and text[index + 1] in _CLOSING_MARKS
            ):
                index += 1
                buffer.append(text[index])

            piece = "".join(buffer).strip()
            if piece:
                result.append(piece)

            buffer = []

        index += 1

    piece = "".join(buffer).strip()
    if piece:
        result.append(piece)

    return result


def _join_text(left: str, right: str) -> str:
    """
    拼接两个片段。

    由于切分过程会去掉边界空格，英文文本在需要时补回一个空格。
    同时覆盖 “Hello.” + “World” 这种句末标点情况。
    """
    if not left:
        return right.strip()

    if not right:
        return left.strip()

    left = left.rstrip()
    right = right.lstrip()

    left_is_latin = re.search(
        r"[A-Za-z0-9][”’\"')\]}>。！？!?.,;:]*$",
        left,
    )
    right_is_latin = re.match(
        r"^[“‘\"'(\[<{]*[A-Za-z0-9]",
        right,
    )

    separator = " " if left_is_latin and right_is_latin else ""
    return left + separator + right


def _split_overlong_atom(
    atom: str,
    max_bytes: int,
) -> list[str]:
    """
    单个原子项自身超长时的最终兜底。

    一般只会发生在极长 URL、哈希值或没有任何分隔符的 token 中。
    """
    result: list[str] = []
    buffer = ""

    for char in atom:
        candidate = buffer + char

        if buffer and _utf8_len(candidate) > max_bytes:
            result.append(buffer)
            buffer = char
        else:
            buffer = candidate

    if buffer:
        result.append(buffer)

    return result


def _preferred_boundary(
    tokens: list[str],
    fit_count: int,
    max_bytes: int,
) -> int:
    """
    在已经能够容纳的 token 中选择更自然的切分位置。

    优先级：
        1. 空格；
        2. 中文连接词之前；
        3. 英文单词之后。
    """
    minimum_size = max_bytes * 0.55
    candidates: list[tuple[int, int]] = []
    prefix_size = 0

    for index, token in enumerate(
        tokens[:fit_count],
        start=1,
    ):
        prefix_size += _utf8_len(token)

        # 避免为了寻找语义边界而产生过短片段。
        if prefix_size < minimum_size:
            continue

        next_token = (
            tokens[index]
            if index < len(tokens)
            else ""
        )

        if token.isspace():
            candidates.append((3, index))

        elif next_token in _SOFT_BREAK_WORDS:
            candidates.append((2, index))

        elif re.fullmatch(
            r"""
            [A-Za-z0-9]+
            (?:[’'][A-Za-z0-9]+)*
            (?:-[A-Za-z0-9]+)*
            """,
            token,
            re.VERBOSE,
        ):
            candidates.append((1, index))

    if not candidates:
        return fit_count

    highest_priority = max(
        priority
        for priority, _ in candidates
    )

    return max(
        index
        for priority, index in candidates
        if priority == highest_priority
    )


def _hard_split_core(
    text: str,
    max_bytes: int,
) -> list[str]:
    """
    对无可用标点的长文本进行兜底切分。

    普通英文单词、数字、邮箱、URL 和缩写不会被拆开，
    除非单个原子项本身已经超过 max_bytes。
    """
    tokens = _ATOM_PATTERN.findall(text)
    result: list[str] = []

    while tokens:
        current_size = 0
        fit_count = 0

        # 找出从当前位置开始能够容纳的最大 token 数量。
        while (
            fit_count < len(tokens)
            and current_size + _utf8_len(tokens[fit_count])
            <= max_bytes
        ):
            current_size += _utf8_len(tokens[fit_count])
            fit_count += 1

        if fit_count == len(tokens):
            piece = "".join(tokens).strip()
            if piece:
                result.append(piece)
            break

        if fit_count == 0:
            # 单个 URL 或 token 已经超过限制。
            atom = tokens.pop(0)
            result.extend(
                piece
                for piece in _split_overlong_atom(
                    atom,
                    max_bytes,
                )
                if piece.strip()
            )
            continue

        cut = _preferred_boundary(
            tokens,
            fit_count,
            max_bytes,
        )

        piece = "".join(tokens[:cut]).strip()
        if piece:
            result.append(piece)

        tokens = tokens[cut:]

        # 新片段不以空格开头。
        while tokens and tokens[0].isspace():
            tokens.pop(0)

    return result


def _hard_split(
    text: str,
    max_bytes: int,
) -> list[str]:
    """
    包装无标点兜底切分，并避免句末标点成为独立片段。
    """
    text = _restore_internal_punctuation(text)
    match = _TRAILING_PUNCT_PATTERN.search(text)

    if not match or match.start() == 0:
        return _hard_split_core(text, max_bytes)

    body = text[:match.start()]
    tail = match.group(1)
    tail_size = _utf8_len(tail)

    if tail_size >= max_bytes:
        return _hard_split_core(text, max_bytes)

    chunks = _hard_split_core(body, max_bytes)

    if not chunks:
        return [tail]

    candidate = chunks[-1] + tail

    if _utf8_len(candidate) <= max_bytes:
        chunks[-1] = candidate
        return chunks

    # 为句末标点预留字节，只重切最后一个片段。
    last_chunk = chunks.pop()

    last_parts = _hard_split_core(
        last_chunk,
        max_bytes - tail_size,
    )

    chunks.extend(last_parts[:-1])
    chunks.append(last_parts[-1] + tail)

    return chunks


def _split_oversized_sentence(
    sentence: str,
    max_bytes: int,
) -> list[str]:
    """
    处理超过长度限制的完整句子。

    先按弱停顿标点切分，只有弱标点仍无法满足限制时，
    才进入英文词边界或中文字符边界兜底。
    """
    clauses = _split_by_punctuation(
        sentence,
        _WEAK_PUNCT,
    )

    result: list[str] = []
    buffer = ""

    for clause in clauses:
        if _utf8_len(clause) > max_bytes:
            if buffer:
                result.append(buffer)
                buffer = ""

            result.extend(
                _hard_split(clause, max_bytes)
            )
            continue

        candidate = _join_text(buffer, clause)

        if (
            not buffer
            or _utf8_len(candidate) <= max_bytes
        ):
            buffer = candidate
        else:
            result.append(buffer)
            buffer = clause

    if buffer:
        result.append(buffer)

    return result


def _merge_short_chunks(
    chunks: list[str],
    min_bytes: int,
    max_bytes: int,
) -> list[str]:
    """
    回并过短片段。

    同时尝试与前一个和后一个片段合并，
    选择合并后更紧凑且不超过上限的一侧。
    """
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    index = 0

    while index < len(chunks):
        if (
            _utf8_len(chunks[index]) >= min_bytes
            or len(chunks) == 1
        ):
            index += 1
            continue

        candidates: list[tuple[int, str, str]] = []

        if index > 0:
            merged = _join_text(
                chunks[index - 1],
                chunks[index],
            )

            if _utf8_len(merged) <= max_bytes:
                candidates.append(
                    (_utf8_len(merged), "previous", merged)
                )

        if index + 1 < len(chunks):
            merged = _join_text(
                chunks[index],
                chunks[index + 1],
            )

            if _utf8_len(merged) <= max_bytes:
                candidates.append(
                    (_utf8_len(merged), "next", merged)
                )

        if not candidates:
            index += 1
            continue

        # 合并后越短越优先；相同时优先向前合并。
        _, direction, merged = min(
            candidates,
            key=lambda item: (
                item[0],
                item[1] != "previous",
            ),
        )

        if direction == "previous":
            chunks[index - 1] = merged
            del chunks[index]
            index = max(index - 1, 0)
        else:
            chunks[index] = merged
            del chunks[index + 1]

    return chunks


def _split_one_line(
    protected_text: str,
    max_bytes: int,
    min_bytes: int,
) -> list[str]:
    """切分单行文本，不允许与其他行跨行回并。"""

    sentences = _split_by_punctuation(
        protected_text,
        _STRONG_PUNCT | {"."},
    )

    chunks: list[str] = []
    buffer = ""

    for sentence in sentences:
        if _utf8_len(sentence) > max_bytes:
            if buffer:
                chunks.append(buffer)
                buffer = ""

            chunks.extend(
                _split_oversized_sentence(
                    sentence,
                    max_bytes,
                )
            )
            continue

        candidate = _join_text(buffer, sentence)

        if (
            not buffer
            or _utf8_len(candidate) <= max_bytes
        ):
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = sentence

    if buffer:
        chunks.append(buffer)

    chunks = [
        _restore_internal_punctuation(chunk)
        for chunk in chunks
    ]

    return _merge_short_chunks(
        chunks,
        min_bytes,
        max_bytes,
    )


def smart_split_text(
    text: str,
    max_bytes: int = 90,
    min_bytes: int = 18,
    remove_brackets: bool = False,
) -> list[str]:
    """
    将中英文混合文本切分为适合 F5-TTS 的短文本。

    长度使用 UTF-8 字节数：
        - ASCII 字符通常为 1 字节；
        - 中文字符通常为 3 字节。

    切分优先级：
        1. 换行；
        2. 中英文句末标点；
        3. 中英文逗号、分号、冒号、破折号；
        4. 英文词边界或中文启发式边界；
        5. Unicode 字符级最终兜底。

    Args:
        text:
            待切分文本。

        max_bytes:
            单个片段允许的最大 UTF-8 字节数。默认 90。

        min_bytes:
            短片段回并阈值。
            小于该值的片段会尝试与相邻片段合并。

        remove_brackets:
            是否删除成对括号及其内部内容。

    Returns:
        切分后的非空文本列表。
    """
    if not isinstance(text, str) or not text.strip():
        return []

    if max_bytes <= 0:
        raise ValueError("max_bytes 必须大于 0")

    if min_bytes < 0:
        raise ValueError("min_bytes 不能小于 0")

    if min_bytes > max_bytes:
        raise ValueError(
            "min_bytes 不能大于 max_bytes"
        )

    normalized = _normalize_text(
        text,
        remove_brackets,
    )

    if not normalized:
        return []

    protected = _protect_internal_punctuation(
        normalized
    )

    result: list[str] = []

    # 换行作为硬语义边界，每行独立切分和回并。
    for line in protected.split("\n"):
        line = line.strip()

        if not line:
            continue

        result.extend(
            _split_one_line(
                line,
                max_bytes,
                min_bytes,
            )
        )

    return result


if __name__ == '__main__':
    ai_reply = (
            "- 第一段介绍项目背景，并说明当前进度。\n"
            "## 第二段列出尚未解决的问题，以及接下来的计划。"
            "The third paragraph contains an English summary."
        )
    print(smart_split_text(ai_reply))