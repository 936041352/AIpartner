from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool


# ============================================================
# 类型
# ============================================================

WebSearchProvider = Literal["zhipu", "baidu"]


# ============================================================
# 环境变量
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


# 智谱
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 可以在 .env 覆盖；不设置就使用默认值
ZHIPU_SEARCH_MODEL = os.getenv(
    "ZHIPU_SEARCH_MODEL",
    "glm-4.7-flashx",
)

ZHIPU_SEARCH_ENGINE = os.getenv(
    "ZHIPU_SEARCH_ENGINE",
    "search_pro",
)


# 百度
BAIDU_API_KEY = os.getenv(
    "BAIDU_API_KEY"
)

BAIDU_SEARCH_URL = (
    "https://qianfan.baidubce.com/"
    "v2/ai_search/web_summary"
)


SEARCH_FAIL_PREFIX = "查询失败，失败的原因为："


# ============================================================
# Provider 状态
# ============================================================

_PROVIDER_STATE = {
    "zhipu": {
        "configured": bool(ZHIPU_API_KEY),
        "initialized": False,
        "last_error": None,
        "health_check_attempts": 0,
    },
    "baidu": {
        "configured": bool(BAIDU_API_KEY),
        "initialized": False,
        "last_error": None,
        "health_check_attempts": 0,
    },
}


# 当 provider=None 时：
# 如果已经自动选择成功过，就直接复用。
#
# 注意：
# 这里只记录“自动初始化选择结果”，
# 不影响已经创建好的角色 Tool。
_AUTO_SELECTED_PROVIDER: WebSearchProvider | None = None


# 防止 GUI 中多个角色同时初始化时
# 重复执行健康检查。
_INIT_LOCK = threading.RLock()


# ============================================================
# 通用辅助函数
# ============================================================

def _failure(reason: str) -> str:
    """生成统一的失败文本。"""
    return f"{SEARCH_FAIL_PREFIX}{reason}"


def _is_failure(result: str) -> bool:
    """判断搜索结果是否是失败结果。"""
    return result.startswith(SEARCH_FAIL_PREFIX)


def _normalize_provider(
    provider: str | None,
) -> WebSearchProvider | None:
    """
    标准化 provider 名称。
    """

    if provider is None:
        return None

    provider = provider.strip().lower()

    if provider == "zhipu":
        return "zhipu"

    if provider == "baidu":
        return "baidu"

    raise ValueError(
        "web search provider 只能是 "
        "'zhipu'、'baidu' 或 None"
    )


# ============================================================
# 百度搜索
# ============================================================

def _search_baidu(query: str) -> str:
    """
    百度智能搜索生成（高性能版）。

    成功：
        返回总结后的文本。

    失败：
        返回：
        查询失败，失败的原因为：...
    """
    query = query + "（回复字数不超过400字）"
    if not BAIDU_API_KEY:
        return _failure(
            "未配置 BAIDU_API_KEY"
        )

    headers = {
        "X-Appbuilder-Authorization": (
            f"Bearer {BAIDU_API_KEY}"
        ),
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            BAIDU_SEARCH_URL,
            headers=headers,
            json=payload,
            timeout=(10, 60),
        )

    except requests.exceptions.Timeout:
        return _failure("百度搜索请求超时")

    except requests.exceptions.SSLError as e:
        return _failure(
            f"百度搜索 SSL 连接错误 - {e}"
        )

    except requests.exceptions.ConnectionError as e:
        return _failure(
            f"百度搜索网络连接错误 - {e}"
        )

    except requests.exceptions.RequestException as e:
        return _failure(
            f"百度搜索请求异常 - {e}"
        )

    # --------------------------------------------------------
    # HTTP 错误
    # --------------------------------------------------------

    if not response.ok:
        try:
            error_data = response.json()

            error_message = (
                error_data.get("message")
                or error_data.get("error")
                or response.text
            )

        except ValueError:
            error_message = response.text

        return _failure(
            f"百度搜索 HTTP "
            f"{response.status_code} - "
            f"{error_message}"
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:
        data = response.json()

    except ValueError:
        return _failure(
            "百度搜索返回内容不是有效 JSON"
        )

    # --------------------------------------------------------
    # 百度业务错误
    # --------------------------------------------------------

    if data.get("code"):
        return _failure(
            f"百度搜索 "
            f"{data.get('code')} - "
            f"{data.get('message', '未知错误')}"
        )

    # --------------------------------------------------------
    # 提取答案
    # --------------------------------------------------------

    try:
        answer = (
            data["choices"][0]
            ["message"]["content"]
        )

    except (KeyError, IndexError, TypeError):
        return _failure(
            f"百度搜索返回格式异常 - {data}"
        )

    if not answer:
        return _failure(
            "百度搜索返回内容为空"
        )

    return answer


# ============================================================
# 智谱搜索
# ============================================================

def _search_zhipu(query: str) -> str:
    """
    智谱 Web Search in Chat。

    使用：
        GLM-4.7-FlashX
        + search_pro
        + 关闭 thinking

    模型和搜索引擎可以通过 .env 覆盖。
    """

    if not ZHIPU_API_KEY:
        return _failure(
            "未配置 ZHIPU_API_KEY"
        )

    # Lazy import：
    # 即使机器没有安装 zai-sdk，
    # 百度搜索仍然可以正常使用。
    try:
        from zai import ZhipuAiClient

    except ImportError:
        return _failure(
            "未安装 zai-sdk，"
            "请运行：pip install -U zai-sdk"
        )

    try:
        client = ZhipuAiClient(
            api_key=ZHIPU_API_KEY
        )

        tools = [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": (
                        ZHIPU_SEARCH_ENGINE
                    ),
                    "search_result": True,

                    # 控制一次查询读取的搜索结果数量。
                    # 这里优先兼顾速度与质量。
                    "count": 5,

                    "content_size": "medium",

                    "search_prompt": (
                        "请根据网络搜索结果直接、准确地回答用户的问题。\n"
                        "1. 优先采用最新、可靠的信息；\n"
                        "2. 如果不同来源存在明显冲突，请在回答中指出。\n"
                        "3. 如果用户问问题比较具体，请更多的强调细节；如果用户问问题比较宽泛，请更多的输出概览。\n"
                        "4. 回复的内容不超过500字\n\n"
                        "网络搜索结果：\n"
                        "{search_result}"
                    ),
                },
            }
        ]

        response = (
            client.chat.completions.create(
                model=ZHIPU_SEARCH_MODEL,

                messages=[
                    {
                        "role": "user",
                        "content": query,
                    }
                ],

                tools=tools,

                # 搜索主要追求速度，
                # 不开启额外推理。
                thinking={
                    "type": "disabled"
                },

                stream=False,
            )
        )

    except Exception as e:
        return _failure(
            f"智谱搜索请求异常 - {e}"
        )

    # --------------------------------------------------------
    # 提取答案
    # --------------------------------------------------------

    try:
        answer = (
            response.choices[0]
            .message
            .content
        )

    except (
        AttributeError,
        IndexError,
        TypeError,
    ):
        return _failure(
            f"智谱搜索返回格式异常 - {response}"
        )

    if not answer:
        return _failure(
            "智谱搜索返回内容为空"
        )

    return answer


# ============================================================
# 统一搜索入口
# ============================================================

def web_search(
    query: str,
    provider: str,
) -> str:
    """
    使用指定 provider 搜索互联网。

    参数:
        query:
            查询问题。

        provider:
            "zhipu"
            "baidu"

    返回:
        成功：
            搜索结果。

        失败：
            查询失败，失败的原因为：...
    """

    if not isinstance(query, str):
        return _failure(
            "查询内容必须是字符串"
        )

    query = query.strip()

    if not query:
        return _failure(
            "查询内容不能为空"
        )

    try:
        provider = _normalize_provider(provider)

    except ValueError as e:
        return _failure(str(e))

    if provider == "zhipu":
        return _search_zhipu(query)

    if provider == "baidu":
        return _search_baidu(query)

    return _failure(
        "未指定网络搜索 provider"
    )


# ============================================================
# Provider 健康检查
# ============================================================

def _initialize_provider(
    provider: WebSearchProvider,
) -> tuple[bool, WebSearchProvider | None, str]:
    """
    初始化指定 Provider。

    已经成功初始化过：
        直接返回，不产生 API 调用。

    没有成功初始化过：
        实际调用一次 API 检测。

    失败不会缓存为“永久失败”，
    后续可以再次尝试。
    """

    state = _PROVIDER_STATE[provider]

    # --------------------------------------------------------
    # 没有配置 Key
    # --------------------------------------------------------

    if not state["configured"]:
        return (
            False,
            None,
            f"{provider} 未配置 API Key",
        )

    # --------------------------------------------------------
    # 已经成功验证过
    # --------------------------------------------------------

    if state["initialized"]:
        return (
            True,
            provider,
            f"{provider} 网络搜索已初始化",
        )

    # --------------------------------------------------------
    # 真正健康检查
    # --------------------------------------------------------

    with _INIT_LOCK:

        # 等锁期间可能已经被其他线程初始化。
        if state["initialized"]:
            return (
                True,
                provider,
                f"{provider} 网络搜索已初始化",
            )

        state["health_check_attempts"] += 1

        # 健康检查要尽可能简单，
        # 但明确要求进行联网查询。
        result = web_search(
            query=(
                "请联网查询今天的日期，"
                "只回答年月日。"
            ),
            provider=provider,
        )

        # ----------------------------------------------------
        # 初始化失败
        # ----------------------------------------------------

        if _is_failure(result):

            reason = result.removeprefix(
                SEARCH_FAIL_PREFIX
            )

            state["last_error"] = reason

            # initialized 保持 False。
            # 因此以后还能重新尝试。

            return (
                False,
                None,
                reason,
            )

        # ----------------------------------------------------
        # 初始化成功
        # ----------------------------------------------------

        state["initialized"] = True
        state["last_error"] = None

        return (
            True,
            provider,
            f"{provider} 网络搜索初始化成功",
        )


# ============================================================
# 对外初始化入口
# ============================================================

def initialize_web_search(
    provider: str | None = None,
) -> tuple[bool, WebSearchProvider | None, str]:
    """
    初始化网络搜索。

    规则：

    1. provider 明确指定：
       只使用指定 provider。
       不执行 fallback。

    2. provider=None：
       自动选择。

       优先级：
           智谱
           ↓
           百度

    3. 两个都没有配置：
       初始化失败。

    4. 自动选择成功后：
       本进程后续自动初始化直接复用
       已选 provider，不再产生健康检查。

    返回:
        (
            success,
            selected_provider,
            message,
        )
    """

    global _AUTO_SELECTED_PROVIDER

    # ========================================================
    # 显式指定 provider
    # ========================================================

    if provider is not None:

        try:
            normalized = _normalize_provider(
                provider
            )

        except ValueError as e:
            return (
                False,
                None,
                str(e),
            )

        return _initialize_provider(
            normalized
        )

    # ========================================================
    # 自动选择
    # ========================================================

    with _INIT_LOCK:

        # ----------------------------------------------------
        # 已经自动选择成功过
        # ----------------------------------------------------

        if _AUTO_SELECTED_PROVIDER is not None:

            selected = _AUTO_SELECTED_PROVIDER

            if _PROVIDER_STATE[
                selected
            ]["initialized"]:

                return (
                    True,
                    selected,
                    (
                        f"自动使用已初始化的 "
                        f"{selected} 网络搜索"
                    ),
                )

        # ----------------------------------------------------
        # 没有任何 API Key
        # ----------------------------------------------------

        configured_providers = [
            name
            for name in ("zhipu", "baidu")
            if _PROVIDER_STATE[
                name
            ]["configured"]
        ]

        if not configured_providers:
            return (
                False,
                None,
                (
                    "未配置 ZHIPU_API_KEY 或 "
                    "BAIDU_API_KEY"
                ),
            )

        errors: list[str] = []

        # ----------------------------------------------------
        # 优先智谱，然后百度
        # ----------------------------------------------------

        for candidate in (
            "zhipu",
            "baidu",
        ):

            if not _PROVIDER_STATE[
                candidate
            ]["configured"]:
                continue

            success, selected, message = (
                _initialize_provider(candidate)
            )

            if success and selected:

                # 只缓存自动选择结果。
                #
                # Tool 本身仍然绑定具体 provider，
                # 因此不会导致多角色之间串线。
                _AUTO_SELECTED_PROVIDER = selected

                return (
                    True,
                    selected,
                    (
                        f"自动选择 {selected}："
                        f"{message}"
                    ),
                )

            errors.append(
                f"{candidate}: {message}"
            )

        # ----------------------------------------------------
        # 所有候选均失败
        # ----------------------------------------------------

        return (
            False,
            None,
            (
                "所有网络搜索服务初始化失败；"
                + "；".join(errors)
            ),
        )


# ============================================================
# LangChain Tool Factory
# ============================================================

def build_web_search_tool(
    provider: str,
):
    """
    创建绑定指定 provider 的 LangChain Tool。

    注意：
        这里使用 Factory，而不是全局共享
        CURRENT_PROVIDER。

    因此不同角色可以绑定不同搜索服务，
    不会互相覆盖。
    """

    # 判断 provider in {None, 'zhipu', 'baidu'}
    normalized = _normalize_provider(
        provider
    )

    if normalized is None:
        raise ValueError(
            "创建网络搜索工具时必须指定 provider"
        )

    state = _PROVIDER_STATE[
        normalized
    ]

    if not state["initialized"]:
        raise RuntimeError(
            f"{normalized} 尚未初始化成功，"
            "请先调用 initialize_web_search()"
        )

    # provider 被 closure 固定。
    selected_provider = normalized

    @tool("web_search_tool")
    def web_search_tool(
        query: str,
    ) -> str:
        """
        搜索互联网中的最新或实时信息，并返回整理后的答案。

        当用户的问题涉及今天、当前、最新、近期、实时、
        新闻、政策、价格、市场变化、产品更新、人物现状
        等可能随时间变化的信息时，应优先使用此工具。

        如果用户明确要求联网查询、网络搜索、查找最新信息，
        也应使用此工具。

        Args:
            query:
                需要在互联网中查询的问题。
                应尽量包含完整、明确的检索意图。
        """

        return web_search(
            query=query,
            provider=selected_provider,
        )

    return web_search_tool


# ============================================================
# 状态查询
# ============================================================

def get_web_search_status() -> dict:
    """
    查看当前模块状态。

    不产生任何网络请求。
    不返回 API Key。
    """

    with _INIT_LOCK:
        return {
            "auto_selected_provider": (
                _AUTO_SELECTED_PROVIDER
            ),

            "zhipu": {
                "configured": (
                    _PROVIDER_STATE[
                        "zhipu"
                    ]["configured"]
                ),
                "initialized": (
                    _PROVIDER_STATE[
                        "zhipu"
                    ]["initialized"]
                ),
                "last_error": (
                    _PROVIDER_STATE[
                        "zhipu"
                    ]["last_error"]
                ),
                "health_check_attempts": (
                    _PROVIDER_STATE[
                        "zhipu"
                    ][
                        "health_check_attempts"
                    ]
                ),
                "model": ZHIPU_SEARCH_MODEL,
                "search_engine": (
                    ZHIPU_SEARCH_ENGINE
                ),
            },

            "baidu": {
                "configured": (
                    _PROVIDER_STATE[
                        "baidu"
                    ]["configured"]
                ),
                "initialized": (
                    _PROVIDER_STATE[
                        "baidu"
                    ]["initialized"]
                ),
                "last_error": (
                    _PROVIDER_STATE[
                        "baidu"
                    ]["last_error"]
                ),
                "health_check_attempts": (
                    _PROVIDER_STATE[
                        "baidu"
                    ][
                        "health_check_attempts"
                    ]
                ),
            },
        }