from collections import deque
from threading import Condition, Lock


class DisplayQueueEmptyError(LookupError):
    """当前没有可领取的演出。"""


class DisplayQueueOrderError(RuntimeError):
    """请求的演出不是当前 FIFO 队首。"""


class CharacterDisplayState:
    """保存单个角色当前一轮回复的演出生产和待领取数据。"""

    def __init__(self):
        self._condition = Condition(Lock())
        self._unplanned_queue = deque()
        self._display_queue = deque()
        self._processing = False
        self._input_finished = True
        self._worker_finished = True

        self._turn_id: str | None = None
        self._reply_finished = False
        self._final_reply: str | None = None
        self._turn_error: str | None = None

    def start_turn(self, turn_id: str) -> None:
        """初始化新一轮；上一轮仍有待展示内容时拒绝覆盖。"""
        with self._condition:
            if (
                self._unplanned_queue
                or self._display_queue
                or self._processing
                or not self._worker_finished
            ):
                raise RuntimeError("上一轮演出尚未处理完成。")

            self._turn_id = turn_id
            self._reply_finished = False
            self._final_reply = None
            self._turn_error = None
            self._input_finished = False
            self._worker_finished = False

    def enqueue_unplanned(
        self,
        turn_id: str,
        source_type: str,
        content: str,
    ) -> None:
        """按模型产生回复的顺序加入待解析队列。"""
        content = content.strip()
        if not content:
            return

        with self._condition:
            if turn_id != self._turn_id or self._input_finished:
                raise RuntimeError("演出数据不属于当前活动轮次。")
            self._unplanned_queue.append(
                {
                    "turn_id": turn_id,
                    "type": source_type,
                    "content": content,
                }
            )
            self._condition.notify_all()

    def set_final_reply(self, turn_id: str, final_reply: str) -> None:
        with self._condition:
            if turn_id != self._turn_id:
                return
            self._reply_finished = True
            self._final_reply = final_reply

    def reply_finished(self, turn_id: str) -> bool:
        with self._condition:
            return turn_id == self._turn_id and self._reply_finished

    def finish_input(self, turn_id: str) -> None:
        """声明本轮模型不会再产生新的待演出文本。"""
        with self._condition:
            if turn_id != self._turn_id:
                return
            self._input_finished = True
            self._condition.notify_all()

    def take_unplanned(self, turn_id: str) -> dict | None:
        """供唯一 Worker 阻塞读取下一条待演出文本。"""
        with self._condition:
            while (
                turn_id == self._turn_id
                and not self._unplanned_queue
                and not self._input_finished
            ):
                self._condition.wait()

            if turn_id != self._turn_id:
                return None
            if not self._unplanned_queue:
                self._worker_finished = True
                self._condition.notify_all()
                return None

            self._processing = True
            return self._unplanned_queue.popleft()

    def complete_unplanned(self, turn_id: str) -> None:
        with self._condition:
            if turn_id == self._turn_id:
                self._processing = False
                self._condition.notify_all()

    def append_display(self, turn_id: str, display: dict) -> None:
        with self._condition:
            if turn_id != self._turn_id:
                return
            self._display_queue.append(display)
            self._condition.notify_all()

    def set_turn_error(self, turn_id: str, message: str) -> None:
        with self._condition:
            if turn_id == self._turn_id:
                self._turn_error = message

    def wait_until_worker_finished(self, turn_id: str) -> None:
        with self._condition:
            while turn_id == self._turn_id and not self._worker_finished:
                self._condition.wait()

    def mark_worker_finished(self, turn_id: str) -> None:
        """确保 Worker 异常退出时也能解除后台等待。"""
        with self._condition:
            if turn_id == self._turn_id:
                self._processing = False
                self._worker_finished = True
                self._condition.notify_all()

    def snapshot(self) -> dict:
        """返回状态接口所需的一致快照。"""
        with self._condition:
            ready_display_id = (
                self._display_queue[0]["id"]
                if self._display_queue
                else None
            )
            return {
                "turn_id": self._turn_id,
                "display_exist": bool(
                    self._unplanned_queue
                    or self._display_queue
                    or self._processing
                ),
                "ready_display_id": ready_display_id,
                "reply_finished": self._reply_finished,
                "final_reply": self._final_reply,
                "turn_error": self._turn_error,
            }

    def pop_ready_display(self, display_id: str) -> dict:
        """仅允许按 ID 领取当前 FIFO 队首。"""
        with self._condition:
            if not self._display_queue:
                raise DisplayQueueEmptyError("当前没有可领取的演出。")

            ready_display = self._display_queue[0]
            if ready_display["id"] != display_id:
                raise DisplayQueueOrderError(
                    "请求的演出不是当前待领取的队首。"
                )
            return self._display_queue.popleft()
