from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class CharacterActivityStatus:
    """保存单个角色当前正在执行的任务，供前端定期读取。"""

    generating_reply: bool = False
    planning_display: bool = False
    synthesizing_speech: bool = False
    retrieving_memory: bool = False
    searching_web: bool = False
    consolidating_memory: bool = False
    _lock: Lock = field(default_factory=Lock, repr=False)

    @contextmanager
    def running(self, name: str):
        """在任务执行期间置为 True，并确保结束或异常时恢复。"""
        with self._lock:
            setattr(self, name, True)
        try:
            yield
        finally:
            with self._lock:
                setattr(self, name, False)

    def snapshot(self) -> dict[str, bool]:
        """返回适合状态接口传输的一致快照。"""
        with self._lock:
            return {
                "generating_reply": self.generating_reply,
                "planning_display": self.planning_display,
                "synthesizing_speech": self.synthesizing_speech,
                "retrieving_memory": self.retrieving_memory,
                "searching_web": self.searching_web,
                "consolidating_memory": self.consolidating_memory,
            }
