from collections.abc import Callable
from multiprocessing import queues
import time
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp

from gomoku.core.gomoku import Gomoku
from gomoku.model.policy_value_net import PolicyValueNet
from gomoku.utils.config.loader import RootConfig


class BatchInferenceServer:
    """멀티프로세스 환경에서 동적 배칭으로 추론을 처리하는 서버."""

    def __init__(
        self,
        cfg: RootConfig,
        request_q: mp.Queue,
        result_queues: list[mp.Queue],
        model_factory: Callable[[Gomoku, RootConfig], PolicyValueNet] | None = None,
        max_wait_ms: int | None = None,
    ):
        self.cfg = cfg
        self.request_q = request_q
        self.result_queues = result_queues
        self.model_factory = model_factory

        self.batch_size = max(int(cfg.mcts.batch_infer_size), 1)
        # config에 없으면 0으로 즉시 처리; None/음수는 0으로 처리
        cfg_wait = getattr(cfg.mcts, "max_batch_wait_ms", 0) or 0
        self.max_wait_ms = cfg_wait if max_wait_ms is None else max_wait_ms

        self.model: PolicyValueNet | None = None
        self.device = torch.device("cpu")
        self.running = True

    def _init_model(self) -> None:
        """프로세스 시작 후 모델을 초기화한다."""
        game = Gomoku(self.cfg.board)
        if self.model_factory:
            self.model = self.model_factory(game, self.cfg)
        else:
            self.model = PolicyValueNet(game, self.cfg.model, device="cpu")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.model.eval()

    def _collect_batch(self, buf: list[tuple[int, int, Any]]) -> None:
        """목표 배치 사이즈 또는 타임아웃까지 요청을 수집한다."""
        start = time.time()
        max_wait = self.max_wait_ms / 1000.0 if self.max_wait_ms > 0 else 0.0

        while len(buf) < self.batch_size:
            elapsed = time.time() - start
            remaining = max_wait - elapsed

            if buf and remaining <= 0:
                break

            try:
                if not buf and remaining <= 0:
                    item = self.request_q.get()
                elif remaining > 0:
                    item = self.request_q.get(timeout=remaining)
                else:
                    item = self.request_q.get_nowait()
            except queues.Empty:
                break
            except Exception:
                break

            if item is None:
                self.running = False
                break
            buf.append(item)

    def _send_error(
        self, worker_ids: tuple[int, ...], req_ids: tuple[int, ...], error_msg: str
    ) -> None:
        """배치 내 모든 요청자에게 에러 응답을 전파한다."""
        for idx, wid in enumerate(worker_ids):
            try:
                self.result_queues[int(wid)].put(
                    (int(req_ids[idx]), None, None, error_msg)
                )
            except Exception:
                # 큐가 닫혔으면 무시
                pass

    def _process_batch(self, batch: list[tuple[int, int, Any]]) -> None:
        """수집된 배치를 모델에 태우고 결과를 분배한다."""
        if not batch or self.model is None:
            return

        worker_ids, req_ids, states_list = zip(*batch)
        try:
            states_np = np.stack(
                [
                    s.cpu().numpy() if isinstance(s, torch.Tensor) else np.asarray(s)
                    for s in states_list
                ]
            )
        except ValueError as exc:
            shapes = [getattr(s, "shape", None) for s in states_list]
            print(f"[BatchInferenceServer] State shape mismatch: {shapes}")
            self._send_error(worker_ids, req_ids, f"State shape mismatch: {exc}")
            return
        except Exception as exc:  # pragma: no cover - 보호용
            print(f"[BatchInferenceServer] Preprocess error: {exc}")
            self._send_error(worker_ids, req_ids, f"Preprocess error: {exc}")
            return

        # MPInferenceClient는 (1, C, H, W) 형태를 보낼 수 있으므로 squeeze
        if states_np.ndim == 5 and states_np.shape[1] == 1:
            states_np = states_np.reshape(states_np.shape[0], *states_np.shape[2:])

        try:
            inputs = torch.from_numpy(states_np).to(
                self.device, non_blocking=True, dtype=torch.float32
            )

            with torch.inference_mode():
                policy_logits, values = self.model(inputs)

            if policy_logits.dim() == 3:
                policy_logits = policy_logits.unsqueeze(0)
            if values.dim() == 1:
                values = values.unsqueeze(0)

            policies = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()
            values_np = values.view(-1).detach().cpu().numpy()

            for idx, wid in enumerate(worker_ids):
                self.result_queues[int(wid)].put(
                    (
                        int(req_ids[idx]),
                        policies[idx],
                        values_np[idx],
                        None,
                    )
                )
        except Exception as exc:
            print(f"[BatchInferenceServer] Critical Error: {exc}")
            self._send_error(worker_ids, req_ids, f"Inference error: {exc}")

    def run(self) -> None:
        """서버 메인 루프."""
        self._init_model()
        pending: list[tuple[int, int, Any]] = []

        while self.running:
            self._collect_batch(pending)
            if pending:
                self._process_batch(pending)
                pending.clear()
