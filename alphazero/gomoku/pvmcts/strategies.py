import time

import ray
import torch
import torch.nn.functional as F

from gomoku.pvmcts.search.ray.batch_manager import PendingNodeInfo

from .treenode import Node

__all__ = ["SearchStrategiesMixin"]


class SearchStrategiesMixin:
    """
    Mixin that hosts the concrete search variants (non-batch, sync batch, async batch).
    Assumes the consumer defines: game, mcts_params, inference, device, use_fp16,
    batch_size, min_batch_size, max_wait_ns, async_inflight_limit,
    _apply_dirichlet_noise_to_policy, and _infer_single_state.
    """

    def _apply_batch_results(
        self,
        policy_logits: torch.Tensor,
        values: torch.Tensor,
        mapping: list[PendingNodeInfo],
        result_device: torch.device,
    ):
        values = values.unsqueeze(-1) if values.dim() == 0 else values
        values = values.squeeze()
        values = values.float()

        for i in range(len(mapping)):
            node_info = mapping[i]
            node = node_info.node
            is_start_node = node_info.is_start_node
            legal_mask = node.get_legal_mask_tensor(result_device)

            logits = policy_logits[i].float()
            logits_masked = torch.where(
                legal_mask,
                logits,
                torch.tensor(-float("inf"), device=result_device, dtype=logits.dtype),
            )
            policy = F.softmax(logits_masked, dim=-1)

            if is_start_node:
                policy = self._apply_dirichlet_noise_to_policy(policy, legal_mask)

            value_item = (
                values.item() if values.dim() == 0 and i == 0 else values[i].item()
            )

            node.expand_with_policy(policy.cpu(), self.game)
            node.backup(value_item)

    def _run_non_batch_simulations(self, root: Node):
        """Single-state inference per leaf (CPU-focused path)."""
        num_searches = int(self.mcts_params.num_searches)
        encode_dtype = torch.float32

        if root.visit_count == 0:
            value_term_root, is_term_root = root.get_terminal_info()
            if is_term_root:
                root.backup(value_term_root)
                return
            try:
                state_tensor = root.get_encoded_state_tensor(
                    self.game, dtype=encode_dtype, device=self.device
                )
                policy_logits, value_tensor, value_shape = self._infer_single_state(
                    state_tensor
                )

                if value_tensor.numel() == 1:
                    value = value_tensor.item()
                else:
                    print(
                        f"Warning: Unexpected value tensor shape from root infer(): {value_shape}. Using 0.0."
                    )
                    value = 0.0

                legal_mask = root.get_legal_mask_tensor(policy_logits.device)
                logits_masked = torch.where(
                    legal_mask,
                    policy_logits,
                    torch.tensor(
                        -float("inf"),
                        device=policy_logits.device,
                        dtype=policy_logits.dtype,
                    ),
                )
                policy = F.softmax(logits_masked, dim=-1)
                policy = self._apply_dirichlet_noise_to_policy(policy, legal_mask)
                root.expand_with_policy(policy.cpu(), self.game)
                root.backup(value)
            except Exception as e:
                print(f"Error during root node inference (non-batch): {e}")
                root.visit_count += 1
                return

        for _ in range(max(0, num_searches - root.visit_count)):
            node = root
            while node.is_fully_expanded():
                if node.is_terminal():
                    break
                node = node.select()

            value, is_terminal = node.get_terminal_info()

            if not is_terminal and node.visit_count == 0:
                try:
                    state_tensor = node.get_encoded_state_tensor(
                        self.game, dtype=encode_dtype, device=self.device
                    )
                    policy_logits, value_tensor, value_shape = self._infer_single_state(
                        state_tensor
                    )

                    if value_tensor.numel() == 1:
                        value = value_tensor.item()
                    else:
                        print(
                            f"Warning: Unexpected value tensor shape from leaf infer(): {value_shape}. Using 0.0."
                        )
                        value = 0.0

                    legal_mask = node.get_legal_mask_tensor(policy_logits.device)
                    logits_masked = torch.where(
                        legal_mask,
                        policy_logits,
                        torch.tensor(
                            -float("inf"),
                            device=policy_logits.device,
                            dtype=policy_logits.dtype,
                        ),
                    )
                    policy = F.softmax(logits_masked, dim=-1)
                    node.expand_with_policy(policy.cpu(), self.game)

                except Exception as e:
                    print(
                        f"Error during non-batch inference/expansion for node {node.action_taken}: {e}"
                    )
                    value = 0.0

            node.backup(value)

    def _run_sync_batch_simulations(self, root: Node):
        """Synchronous batched inference (local GPU or sync remote)."""
        sim_count = 0
        num_searches = int(self.mcts_params.num_searches)
        nodes_in_flight: dict[Node, bool] = {}
        encode_dtype = torch.float16 if self.use_fp16 else torch.float32

        if root.visit_count == 0:
            value_term_root, is_term_root = root.get_terminal_info()
            if is_term_root:
                root.backup(value_term_root)
                return
            root_encode = root.get_encoded_state_tensor(
                self.game, dtype=encode_dtype, device=self.device
            )
            try:
                policy_logits, values = self.inference.infer_batch(
                    root_encode.unsqueeze(0)
                )
                result_device = policy_logits.device
                root_mapping = [PendingNodeInfo(node=root, is_start_node=True)]
                self._apply_batch_results(
                    policy_logits, values, root_mapping, result_device
                )
                sim_count += 1
            except Exception as e:
                print(f"Error during root node inference (sync-batch): {e}")
                root.visit_count += 1
                return

        while sim_count < num_searches:
            pending_batch_nodes: list[Node] = []
            pending_batch_info: list[PendingNodeInfo] = []

            while (
                len(pending_batch_nodes) < self.batch_size
                and sim_count + len(pending_batch_nodes) < num_searches
            ):
                node = root
                is_start_node = True
                while node.is_fully_expanded():
                    is_start_node = False
                    next_node = node.select()
                    value_term, is_term = next_node.get_terminal_info()
                    if next_node in nodes_in_flight or is_term:
                        if is_term and next_node not in nodes_in_flight:
                            next_node.backup(value_term)
                            nodes_in_flight[next_node] = True
                        node = None
                        break
                    node = next_node

                if node is None:
                    continue

                if node.visit_count == 0 and node not in nodes_in_flight:
                    pending_batch_nodes.append(node)
                    pending_batch_info.append(
                        PendingNodeInfo(node=node, is_start_node=is_start_node)
                    )
                    nodes_in_flight[node] = True

            if pending_batch_nodes:
                batch_states_list = [
                    p_node.get_encoded_state_tensor(
                        self.game, dtype=encode_dtype, device=self.device
                    )
                    for p_node in pending_batch_nodes
                ]
                batch_tensor = torch.stack(batch_states_list)

                try:
                    policy_logits, values = self.inference.infer_batch(batch_tensor)
                    result_device = policy_logits.device
                    self._apply_batch_results(
                        policy_logits, values, pending_batch_info, result_device
                    )
                    sim_count += len(pending_batch_nodes)
                except Exception as e:
                    print(f"Warning: Batch inference failed (sync): {e}")
                    for p_info in pending_batch_info:
                        p_info.node.visit_count += 1
                for node in pending_batch_nodes:
                    nodes_in_flight.pop(node, None)
            else:
                break

    def _run_async_batch_simulations(self, root: Node):
        """Asynchronous batched inference (Ray GPU path)."""
        pending_refs: dict[ray.ObjectRef, list[PendingNodeInfo]] = {}
        sim_count = 0
        num_searches = int(self.mcts_params.num_searches)
        nodes_in_flight: dict[Node, bool] = {}
        encode_dtype = torch.float16 if self.use_fp16 else torch.float32
        max_inflight = self.async_inflight_limit

        pending_batch: list[PendingNodeInfo] = []
        batch_start_time_ns: int | None = None
        stall_rounds = 0

        def submit_batch(batch_info: list[PendingNodeInfo]) -> None:
            nonlocal batch_start_time_ns
            if not batch_info:
                return
            batch_states = [
                item.node.get_encoded_state_tensor(
                    self.game, dtype=encode_dtype, device=self.device
                )
                for item in batch_info
            ]
            batch_tensor = torch.stack(batch_states)
            try:
                ref = self.inference.infer_async(batch_tensor)
                pending_refs[ref] = batch_info
                if pending_batch:
                    batch_start_time_ns = time.monotonic_ns()
                else:
                    batch_start_time_ns = None
            except Exception as exc:
                print(f"Error submitting inference batch (async): {exc}")
                for info in batch_info:
                    info.node.visit_count += 1
                    nodes_in_flight.pop(info.node, None)

        def drain_ready(timeout_s: float = 0.0) -> int:
            if not pending_refs:
                return 0
            ready_refs, _ = ray.wait(
                list(pending_refs.keys()),
                num_returns=len(pending_refs),
                timeout=timeout_s,
            )
            if not ready_refs:
                return 0

            processed = 0
            for ref in ready_refs:
                mapping = pending_refs.pop(ref, [])
                if not mapping:
                    continue
                try:
                    policy_logits, values = ray.get(ref)
                    result_device = policy_logits.device
                    self._apply_batch_results(
                        policy_logits, values, mapping, result_device
                    )
                    processed += len(mapping)
                except Exception as exc:
                    print(f"Warning: Ray inference failed for a batch (async): {exc}")
                    for info in mapping:
                        info.node.visit_count += 1
                finally:
                    for info in mapping:
                        nodes_in_flight.pop(info.node, None)
            return processed

        root_value, root_is_terminal = root.get_terminal_info()
        if root_is_terminal:
            root.backup(root_value)
            return

        root_tensor = root.get_encoded_state_tensor(
            self.game, dtype=encode_dtype, device=self.device
        ).unsqueeze(0)
        try:
            root_ref = self.inference.infer_async(root_tensor)
            pending_refs[root_ref] = [PendingNodeInfo(node=root, is_start_node=True)]
            nodes_in_flight[root] = True
        except Exception as exc:
            print(f"Error submitting root inference (async): {exc}")
            root.visit_count += 1
            return

        def inflight_simulations() -> int:
            return sum(len(mapping) for mapping in pending_refs.values()) + len(
                pending_batch
            )

        while sim_count < num_searches:
            progress_made = False

            while (
                len(pending_batch) < self.batch_size
                and sim_count + inflight_simulations() < num_searches
                and (
                    max_inflight is None
                    or len(pending_refs) + (1 if pending_batch else 0) < max_inflight
                )
            ):
                node = root
                is_start_node = True

                while node.is_fully_expanded():
                    is_start_node = False
                    if node is root and not root.children:
                        node = None
                        break
                    next_node = node.select()
                    value_term, is_term = next_node.get_terminal_info()
                    if next_node in nodes_in_flight or is_term:
                        if is_term and next_node not in nodes_in_flight:
                            next_node.backup(value_term)
                            nodes_in_flight[next_node] = True
                        node = None
                        break
                    node = next_node

                if node is None:
                    break

                if node.visit_count == 0 and node not in nodes_in_flight:
                    pending_batch.append(
                        PendingNodeInfo(node=node, is_start_node=is_start_node)
                    )
                    nodes_in_flight[node] = True
                    progress_made = True
                    if batch_start_time_ns is None:
                        batch_start_time_ns = time.monotonic_ns()

                if not pending_batch:
                    break

                elapsed_ns = (
                    0
                    if batch_start_time_ns is None
                    else time.monotonic_ns() - batch_start_time_ns
                )
                if len(pending_batch) >= self.batch_size or (
                    len(pending_batch) >= self.min_batch_size
                    and elapsed_ns >= self.max_wait_ns
                ):
                    break

            if not pending_batch:
                batch_start_time_ns = None

            if pending_batch and (
                max_inflight is None or len(pending_refs) < max_inflight
            ):
                submit_size = min(len(pending_batch), self.batch_size)
                batch_info = pending_batch[:submit_size]
                pending_batch = pending_batch[submit_size:]
                submit_batch(batch_info)
                progress_made = True
                continue

            processed = drain_ready(0.0)
            if processed:
                sim_count += processed
                progress_made = True
            elif pending_refs:
                processed = drain_ready(0.002)
                if processed:
                    sim_count += processed
                    progress_made = True

            if progress_made:
                stall_rounds = 0
                continue

            stall_rounds += 1
            if not pending_refs and not pending_batch:
                break
            if stall_rounds > 6:
                processed = drain_ready(0.05)
                if processed:
                    sim_count += processed
                    stall_rounds = 0
                    continue
                if not pending_refs and not pending_batch:
                    break

        if pending_refs:
            processed_final = drain_ready(2.0)
            sim_count += processed_final
            if pending_refs:
                for ref, mapping in list(pending_refs.items()):
                    try:
                        policy_logits, values = ray.get(ref, timeout=10.0)
                        result_device = policy_logits.device
                        self._apply_batch_results(
                            policy_logits, values, mapping, result_device
                        )
                        sim_count += len(mapping)
                    except Exception as exc:
                        print(
                            f"Warning: Final processing failed for a batch (async): {exc}"
                        )
                        for info in mapping:
                            info.node.visit_count += 1
                    finally:
                        for info in mapping:
                            nodes_in_flight.pop(info.node, None)
                    pending_refs.pop(ref, None)
