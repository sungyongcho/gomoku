from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import pytest
import torch

from gomoku.inference.mp_client import MPInferenceClient


def _mp_worker_loop(in_q, out_q, action_size: int):
    while True:
        item = in_q.get()
        if item is None:
            break
        worker_id, req_id, tensor = item
        policy = torch.zeros((action_size,), dtype=torch.float32) + worker_id
        value = torch.tensor([float(worker_id)], dtype=torch.float32)
        out_q.put((req_id, policy.numpy(), value.numpy(), None))


def test_mp_ipc_concurrent_infer() -> None:
    """SimpleQueue 기반 MPInferenceClient로 멀티 워커 동시 요청을 검증한다."""
    action_size = 4
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue()
    out_q = ctx.Queue()

    workers = [
        ctx.Process(target=_mp_worker_loop, args=(in_q, out_q, action_size))
        for _ in range(2)
    ]
    for p in workers:
        p.start()

    client0 = MPInferenceClient(worker_id=0, in_q=in_q, out_q=out_q)
    client1 = MPInferenceClient(worker_id=1, in_q=in_q, out_q=out_q)

    state = torch.zeros((1, 2, 2), dtype=torch.float32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut0 = executor.submit(client0.infer, state)
        fut1 = executor.submit(client1.infer, state)
        pol0, val0 = fut0.result()
        pol1, val1 = fut1.result()

    print("mp ipc policies:", pol0, pol1)
    print("mp ipc values:", val0, val1)
    assert float(val0.view(-1)[0]) == pytest.approx(0.0)
    assert float(val1.view(-1)[0]) == pytest.approx(1.0)
    assert pol0.shape[1] == action_size
    assert pol1.shape[1] == action_size

    for _ in workers:
        in_q.put(None)
    for p in workers:
        p.join(timeout=5)
