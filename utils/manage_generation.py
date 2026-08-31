from dataclasses import dataclass
from threading import Lock
from time import time
from uuid import uuid4


class GenerationBusyError(RuntimeError):
    """另一个角色正在占用全局回复生成资源。"""

    def __init__(self, active_job: "GenerationJob"):
        self.active_job = active_job
        super().__init__(
            f"角色“{active_job.character_name}”正在"
            f"{active_job.stage}，请稍后再试。"
        )


@dataclass(frozen=True)
class GenerationJob:
    token: str
    character_name: str
    stage: str
    started_at: float


class GenerationCoordinator:
    """保证整个进程同一时间只执行一个角色生成任务。"""

    def __init__(self):
        self._lock = Lock()
        self._active_job: GenerationJob | None = None

    def try_start(
        self,
        character_name: str,
        stage: str = "生成回复",
    ) -> GenerationJob:
        """原子地登记任务；已有任务时立即拒绝，不阻塞请求线程。"""
        with self._lock:
            if self._active_job is not None:
                raise GenerationBusyError(self._active_job)

            job = GenerationJob(
                token=uuid4().hex,
                character_name=character_name,
                stage=stage,
                started_at=time(),
            )
            self._active_job = job
            return job

    def update_stage(self, token: str, stage: str) -> None:
        with self._lock:
            if self._active_job is None or self._active_job.token != token:
                return
            self._active_job = GenerationJob(
                token=token,
                character_name=self._active_job.character_name,
                stage=stage,
                started_at=self._active_job.started_at,
            )

    def finish(self, token: str) -> None:
        """只允许任务持有者释放状态，避免旧任务误释放新任务。"""
        with self._lock:
            if self._active_job is not None and self._active_job.token == token:
                self._active_job = None

    def status(self) -> GenerationJob | None:
        with self._lock:
            return self._active_job
