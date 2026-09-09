"""
将检索到的记忆转换为相对当前时间的“相对时间”
"""

from datetime import datetime, timedelta
import math


_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)

# 时段划分：[起始小时, 文本]
_DAY_PERIODS = (
    (18, "晚上"),
    (14, "下午"),
    (12, "中午"),
    (9, "上午"),
    (6, "早上"),
    (0, "凌晨"),
)

_WEEKDAY_NAMES = ("一", "二", "三", "四", "五", "六", "日")


def _print_error(message: str) -> None:
    """统一打印错误信息。"""
    print(f"[format_relative_time] {message}！！！")


def _parse_time(value) -> datetime | None:
    """
    支持两种时间格式：
    1. Unix timestamp，如 1756969200、1756969200.5
    2. "%Y-%m-%d %H:%M:%S" 格式的字符串
    """
    # bool 是 int 的子类，但不应被视为合法时间戳
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            _print_error(f"无效的时间戳: {value!r}")
            return None

        try:
            return datetime.fromtimestamp(value)
        except (ValueError, OSError, OverflowError):
            _print_error(f"无效的时间戳: {value!r}")
            return None

    if isinstance(value, str):
        for fmt in _TIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        _print_error(
            f"无效的时间文本: {value!r}，"
            f"支持格式: {', '.join(_TIME_FORMATS)}"
        )
        return None

    _print_error(
        f"不支持的时间类型: {type(value).__name__}，"
        "仅支持 Unix timestamp 或标准时间文本"
    )
    return None


def _get_day_period(dt: datetime) -> str:
    """返回时间所属的自然语言时段。"""
    for start_hour, name in _DAY_PERIODS:
        if dt.hour >= start_hour:
            return name
    return "凌晨"


def _get_week_start(dt: datetime):
    """返回当前日期所在自然周的周一。"""
    return dt.date() - timedelta(days=dt.weekday())


def _get_month_period(dt: datetime) -> str:
    """返回日期在月份中的大致阶段。"""
    if dt.day <= 10:
        return "初"
    if dt.day <= 20:
        return "中旬"
    return "末"


def _is_previous_month(target: datetime, now: datetime) -> bool:
    """判断 target 是否位于 now 的上一个自然月。"""
    if now.month == 1:
        return target.year == now.year - 1 and target.month == 12

    return target.year == now.year and target.month == now.month - 1


def format_relative_time(timestamp, now: datetime | None = None) -> str:
    """
    将时间转换为简短的中文相对时间描述。

    timestamp 支持：
        1. Unix timestamp
        2. "%Y-%m-%d %H:%M:%S" 格式字符串
        3. "%Y/%m/%d %H:%M:%S" 格式字符串

    示例：
        今天上午
        昨天下午
        前天晚上
        本周一早上
        上周五晚上
        本月15号
        上月28号
        今年3月初
        去年11月中旬
        3年前
        2010年

    输入异常或时间晚于当前时间时，打印错误信息并返回空字符串。
    """
    target = _parse_time(timestamp)
    if target is None:
        return ""

    now = now or datetime.now()

    # 当前暂不处理时区，因此未来时间视为异常数据
    if target > now:
        _print_error(
            f"记录时间晚于当前时间: "
            f"{target} > {now.strftime(_TIME_FORMATS[0])}"
        )
        return ""

    period = _get_day_period(target)
    day_diff = (now.date() - target.date()).days

    # 今天 / 昨天 / 前天
    if day_diff == 0:
        return f"今天{period}"
    if day_diff == 1:
        return f"昨天{period}"
    if day_diff == 2:
        return f"前天{period}"

    # 本周 / 上周
    now_week_start = _get_week_start(now)
    target_week_start = _get_week_start(target)
    weekday = _WEEKDAY_NAMES[target.weekday()]

    if target_week_start == now_week_start:
        return f"本周{weekday}{period}"

    if target_week_start == now_week_start - timedelta(days=7):
        return f"上周{weekday}{period}"

    # 本月 / 上月
    if target.year == now.year and target.month == now.month:
        return f"本月{target.day}号"

    if _is_previous_month(target, now):
        return f"上月{target.day}号"

    # 今年 / 去年
    month_period = _get_month_period(target)
    if target.year == now.year:
        return f"今年{target.month}月{month_period}"
    if target.year == now.year - 1:
        return f"去年{target.month}月{month_period}"

    # 更早
    year_diff = now.year - target.year

    if 2 <= year_diff <= 10:
        return f"{year_diff}年前"

    return f"{target.year}年"


if __name__ == "__main__":
    now = datetime(2026, 9, 4, 15, 30, 0)  # 2026-09-04，周五
    print(format_relative_time("2026-09-04 09:30:00", now))
    # 今天上午

    print(format_relative_time("2026-09-04 06:30:00", now))
    # 今天早上

    print(format_relative_time("2026-09-04 00:30:00", now))
    # 今天凌晨

    print(format_relative_time("2026-09-03 15:30:00", now))
    # 昨天下午

    print(format_relative_time("2026-09-02 20:30:00", now))
    # 前天晚上

    print(format_relative_time("2026-09-01 09:30:00", now))
    # 本周二上午

    print(format_relative_time("2026-08-31 06:30:00", now))
    # 本周一早上

    print(format_relative_time("2026-08-30 10:30:00", now))
    # 上周日上午

    print(format_relative_time("2026-08-28 20:30:00", now))
    # 上周五晚上

    print(format_relative_time("2026-08-15 12:30:00", now))
    # 上月15号

    print(format_relative_time("2026-07-31 23:30:00", now))
    # 今年7月

    print(format_relative_time("2026-06-10 09:30:00", now))
    # 今年6月

    print(format_relative_time("2026-01-01 09:30:00", now))
    # 今年1月

    print(format_relative_time("2025-12-31 09:30:00", now))
    # 去年12月

    print(format_relative_time("2025-03-15 09:30:00", now))
    # 去年3月

    print(format_relative_time("2024-09-04 09:30:00", now))
    # 2年前

    print(format_relative_time("2020-09-04 09:30:00", now))
    # 6年前

    print(format_relative_time("2016-09-04 09:30:00", now))
    # 10年前

    print(format_relative_time("2015-09-04 09:30:00", now))
    # 2015年