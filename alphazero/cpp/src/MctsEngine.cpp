#include "MctsEngine.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <map>

namespace {
constexpr int kWinLength = 5;

int CountNodes(const MctsNode *node)
{
	if (node == nullptr) {
		return 0;
	}
	int count = 1;
	for (const auto *child : node->children) {
		count += CountNodes(child);
	}
	return count;
}

bool SameState(const GomokuState &a, const GomokuState &b)
{
	return a.next_player == b.next_player
		&& a.last_move_idx == b.last_move_idx
		&& a.p1_pts == b.p1_pts
		&& a.p2_pts == b.p2_pts
		&& a.empty_count == b.empty_count
		&& a.board == b.board
		&& a.history == b.history;
}
}  // namespace

MctsNode::~MctsNode()
{
	for (auto *child : children) {
		delete child;
	}
}

MctsEngine::MctsEngine(GomokuCore *core, float c_puct, bool fpu_from_parent, float fpu_reduction)
	: core_(core)
	, board_size_(core != nullptr ? core->board_size() : 0)
	, action_size_(core != nullptr ? core->action_size() : 0)
	, c_puct_(c_puct)
	, fpu_from_parent_(fpu_from_parent)
	, fpu_reduction_(fpu_reduction)
{
	if (core_ == nullptr) {
		throw std::invalid_argument("GomokuCore pointer is null");
	}
	if (board_size_ <= 0 || action_size_ <= 0) {
		throw std::invalid_argument("Invalid board_size or action_size");
	}
}

MctsEngine::~MctsEngine()
{
	ClearTree();
}

void MctsEngine::ClearTree()
{
	delete root_;
	root_ = nullptr;
	node_count_ = 0;
}

void MctsEngine::ResetRoot(const GomokuState &root_state)
{
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	ClearTree();
	root_ = new MctsNode();
	root_->player_to_play = root_state.next_player;
	root_state_ = root_state;
	node_count_ = 1;
}

std::vector<std::pair<int, int>> MctsEngine::ChildVisits() const
{
	std::vector<std::pair<int, int>> visits;
	if (root_ == nullptr) {
		return visits;
	}
	visits.reserve(root_->moves.size());
	for (std::size_t i = 0; i < root_->moves.size(); ++i) {
		const MctsNode *child = root_->children.empty() ? nullptr : root_->children[i];
		visits.emplace_back(static_cast<int>(root_->moves[i]), child != nullptr ? child->visit_count : 0);
	}
	return visits;
}

void MctsEngine::AdvanceToChild(std::size_t index, GomokuState next_state)
{
	MctsNode *child = root_->children.empty() ? nullptr : root_->children[index];
	if (child == nullptr) {
		child = new MctsNode();
		child->move_idx = static_cast<int>(root_->moves[index]);
		child->prior = root_->priors[index];
		child->player_to_play = 3 - root_->player_to_play;
	} else {
		root_->children[index] = nullptr;
	}
	delete root_;
	child->parent = nullptr;
	root_ = child;
	root_state_ = std::move(next_state);
	node_count_ = CountNodes(root_);
}

bool MctsEngine::AdvanceRoot(int move_idx)
{
	if (root_ == nullptr) {
		return false;
	}
	for (std::size_t i = 0; i < root_->moves.size(); ++i) {
		if (static_cast<int>(root_->moves[i]) != move_idx) {
			continue;
		}
		AdvanceToChild(
			i,
			core_->apply_move(root_state_, move_idx % board_size_, move_idx / board_size_, root_->player_to_play)
		);
		return true;
	}
	return false;
}

int MctsEngine::AlignRoot(const GomokuState &target)
{
	if (root_ == nullptr) {
		return -1;
	}
	if (SameState(root_state_, target)) {
		return root_->visit_count;
	}
	const int move_idx = target.last_move_idx;
	for (std::size_t i = 0; i < root_->moves.size(); ++i) {
		if (static_cast<int>(root_->moves[i]) != move_idx) {
			continue;
		}
		GomokuState next = core_->apply_move(
			root_state_, move_idx % board_size_, move_idx / board_size_, root_->player_to_play
		);
		if (!SameState(next, target)) {
			return -1;
		}
		AdvanceToChild(i, std::move(next));
		return root_->visit_count;
	}
	return -1;
}

std::vector<std::pair<int, int>> MctsEngine::RunSimulation(
	const GomokuState &root_state,
	int num_simulations,
	EvaluationCallback eval_func
)
{
	// Validate board size roughly
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	if (num_simulations <= 0) {
		return {};
	}
	if (!eval_func) {
		throw std::invalid_argument("eval_func is null");
	}

	ResetRoot(root_state);

	for (int i = 0; i < num_simulations; ++i) {
		GomokuState state = root_state; // Copy
		std::vector<MctsNode *> path;
		path.reserve(64);
		path.push_back(root_);
		MctsNode *leaf = root_;

		while (leaf->is_expanded && !leaf->moves.empty()) {
			leaf = Select(leaf, state);
			path.push_back(leaf);
		}

		// Re-using Core logic for termination check
		// Note: GomokuState has last_move_idx
		bool win_found = false;
		int last_move_idx = state.last_move_idx;
		int winner = 0;

		if (last_move_idx >= 0) {
			// Convert index to x, y
			int x = last_move_idx % board_size_;
			int y = last_move_idx / board_size_;
			if (core_->check_win(state, x, y)) {
				// check_win checks if the LAST MOVE made a win
				// The player who made last_move is (state.next_player's opponent)
				// Wait, core_->check_win logic uses state.board and get_pos(x,y).
				// If p1 just played, state.next_player is p2.
				// The stone at x,y is p1.
				// core_->check_win checks 5-in-row for stone at x,y.
				// So it returns true if p1 (prev player) won.
				win_found = true;
				// Winner is the one who played last move (3 - state.next_player ideally)
				int stone = state.board[last_move_idx]; // Should match
				winner = stone;
			}
		}

		if (!win_found && state.empty_count <= 0) {
			winner = 3; // Draw
		}

		float value = 0.0f;
		if (win_found) {
			// Leaf evaluation is from leaf->player_to_play perspective.
			// If winner == leaf->player_to_play, score +1.
			value = (winner == leaf->player_to_play) ? 1.0f : -1.0f;
		} else if (winner == 3) {
			value = 0.0f;
		} else {
			EvaluationResult eval = eval_func(state);
			const auto &policy = eval.first;
			if (static_cast<int>(policy.size()) != action_size_) {
				throw std::invalid_argument("policy size mismatch");
			}
			value = eval.second;
			ExpandAtLeaf(leaf, state, policy);
		}

		Backup(path, value);
	}

	return ChildVisits();
}

std::vector<std::pair<int, float>> MctsEngine::RootPriors() const
{
	std::vector<std::pair<int, float>> priors;
	if (root_ == nullptr) {
		return priors;
	}
	priors.reserve(root_->moves.size());
	for (std::size_t i = 0; i < root_->moves.size(); ++i) {
		priors.emplace_back(static_cast<int>(root_->moves[i]), root_->priors[i]);
	}
	return priors;
}

std::vector<std::pair<int, int>> MctsEngine::RunSimulationEncoded(
	const GomokuState &root_state,
	int num_simulations,
	const EvaluationCallbackEncoded &eval_func,
	float time_limit_s
)
{
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	if (num_simulations <= 0) {
		return {};
	}
	if (!eval_func) {
		throw std::invalid_argument("eval_func is null");
	}

	ResetRoot(root_state);
	return ContinueSimulationEncoded(num_simulations, eval_func, time_limit_s);
}

std::vector<std::pair<int, int>> MctsEngine::ContinueSimulationEncoded(
	int num_simulations,
	const EvaluationCallbackEncoded &eval_func,
	float time_limit_s,
	int max_nodes,
	bool stoppable
)
{
	if (root_ == nullptr) {
		throw std::logic_error("no tree to continue: call ResetRoot first");
	}
	if (!eval_func) {
		throw std::invalid_argument("eval_func is null");
	}

	last_simulations_ = 0;
	const auto started = std::chrono::steady_clock::now();
	for (int i = 0; i < num_simulations; ++i) {
		if (stoppable && stop_requested_.load(std::memory_order_relaxed)) {
			break;
		}
		if (max_nodes > 0 && node_count_ >= max_nodes) {
			break;
		}
		if (time_limit_s > 0.0f && (i > 0 || root_->is_expanded)) {
			const std::chrono::duration<float> elapsed = std::chrono::steady_clock::now() - started;
			if (elapsed.count() >= time_limit_s) {
				break;
			}
		}

		GomokuState state = root_state_;
		std::vector<MctsNode *> path;
		path.reserve(64);
		path.push_back(root_);
		MctsNode *leaf = root_;

		while (leaf->is_expanded && !leaf->moves.empty()) {
			leaf = Select(leaf, state);
			path.push_back(leaf);
		}

		int last_move = state.last_move_idx;
		bool win_found = false;
		int winner = 0;

		if (last_move >= 0) {
			int x = last_move % board_size_;
			int y = last_move / board_size_;
			if (core_->check_win(state, x, y)) {
				win_found = true;
				winner = state.board[last_move];
			}
		}
		if (!win_found && state.empty_count <= 0) {
			winner = 3;
		}

		float value = 0.0f;
		if (win_found) {
			value = (winner == leaf->player_to_play) ? 1.0f : -1.0f;
		} else if (winner == 3) {
			value = 0.0f;
		} else {
			// Callback needs state, player, last_move
			EvaluationResult eval = eval_func(state, leaf->player_to_play, last_move);
			const auto &policy = eval.first;
			value = eval.second;
			ExpandAtLeaf(leaf, state, policy);
		}

		Backup(path, value);
		++last_simulations_;
		if (leaf == root_ && (win_found || winner == 3)) {
			// The game is over at the root: there is nothing below it to search.
			break;
		}
	}

	return ChildVisits();
}

std::vector<std::pair<int, int>> MctsEngine::RunBatchSimulation(
	const GomokuState &root_state,
	int num_simulations,
	int batch_size,
	EvaluationCallbackBatch eval_func
)
{
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	if (num_simulations <= 0) {
		return {};
	}
	if (!eval_func) {
		throw std::invalid_argument("eval_func is null");
	}
	if (batch_size <= 0) {
		throw std::invalid_argument("batch_size must be positive");
	}

	ResetRoot(root_state);

	int sims_completed = 0;

	// Struct for pending evaluations
	struct PendingItem {
		MctsNode *leaf;
		std::vector<MctsNode *> path;
		GomokuState state;
	};
	std::vector<PendingItem> pending_items;
	pending_items.reserve(batch_size);

	std::vector<GomokuState> states_for_eval;
	states_for_eval.reserve(batch_size);

	while (sims_completed < num_simulations) {
		pending_items.clear();
		states_for_eval.clear();

		// Fill the batch
		while (static_cast<int>(states_for_eval.size()) < batch_size &&
			   sims_completed + static_cast<int>(pending_items.size()) < num_simulations)
		{
			GomokuState state = root_state;
			std::vector<MctsNode *> path;
			path.reserve(64);
			path.push_back(root_);
			MctsNode *leaf = root_;

			// UCB Selection with Virtual Loss
			while (leaf->is_expanded && !leaf->moves.empty()) {
				leaf = Select(leaf, state);
				path.push_back(leaf);
			}

			// Check Terminal
			int last_move = state.last_move_idx;
			bool win_found = false;
			int winner = 0;

			if (last_move >= 0) {
				int x = last_move % board_size_;
				int y = last_move / board_size_;
				if (core_->check_win(state, x, y)) {
					win_found = true;
					winner = state.board[last_move];
				}
			}
			if (!win_found && state.empty_count <= 0) {
				winner = 3;
			}

			if (win_found || winner == 3) {
				// Terminal found - backup immediately (no virtual loss needed)
				float value = 0.0f;
				if (win_found) {
					value = (winner == leaf->player_to_play) ? 1.0f : -1.0f;
				}
				Backup(path, value);
				sims_completed++;
				continue;
			}

            // Check for duplicate leaf in pending evaluations
            bool duplicate = false;
            for (const auto& item : pending_items) {
                if (item.leaf == leaf) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) {
                // Cannot proceed deeper until this leaf is expanded.
                // Stop filling batch and dispatch current items.
                break;
            }

			// Not terminal: Apply Virtual Loss and add to batch
			for (auto *node : path) {
				node->virtual_loss += 1;
			}

			pending_items.push_back({leaf, path, state});
			states_for_eval.push_back(state);
		}

		if (states_for_eval.empty()) {
			if (sims_completed >= num_simulations) break;
			continue;
		}

		// Evaluate Batch
		std::vector<EvaluationResult> results = eval_func(states_for_eval);

		if (results.size() != states_for_eval.size()) {
			throw std::runtime_error("Evaluation callback returned incorrect number of results");
		}

		// Backup Batch
		for (size_t i = 0; i < results.size(); ++i) {
			auto &item = pending_items[i];
			const auto &res = results[i];

			// Revert Virtual Loss
			for (auto *node : item.path) {
				node->virtual_loss -= 1;
			}

			if (static_cast<int>(res.first.size()) != action_size_) {
				throw std::invalid_argument("policy size mismatch");
			}

			ExpandAtLeaf(item.leaf, item.state, res.first);
			Backup(item.path, res.second);
			sims_completed++;
		}
	}

	return ChildVisits();
}

std::vector<std::pair<int, int>> MctsEngine::RunAsyncSimulation(
	const GomokuState &root_state,
	int num_simulations,
	int batch_size,
	AsyncDispatchCallback dispatch_func,
	AsyncCheckCallback check_func,
	int max_inflight_batches
)
{
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	if (max_inflight_batches <= 0) {
		throw std::invalid_argument("max_inflight_batches must be positive");
	}
	if (num_simulations <= 0) return {};
	if (!dispatch_func || !check_func) {
		throw std::invalid_argument("Async callbacks must be provided");
	}
	if (batch_size <= 0) {
		throw std::invalid_argument("batch_size must be positive");
	}

	ResetRoot(root_state);

	int sims_completed = 0;   // Finished (Backed up)
	int sims_dispatched = 0;  // Sent to inference or finished early (terminal)

    // Active Batches
    struct InFlightBatch {
        AsyncHandle handle;
        std::vector<MctsNode*> leaves;
        std::vector<std::vector<MctsNode*>> paths;
        std::vector<GomokuState> states;
    };
    std::map<AsyncHandle, InFlightBatch> in_flight;

    // Current forming batch
    std::vector<MctsNode*> current_leaves;
    std::vector<std::vector<MctsNode*>> current_paths;
    std::vector<GomokuState> current_states;
    current_leaves.reserve(batch_size);
    current_paths.reserve(batch_size);
    current_states.reserve(batch_size);

    while (sims_completed < num_simulations) {
        bool made_progress_selection = false;

        // 1. Selection Phase. Select only while a batch slot is free: leaves picked while
        // earlier results are still out see virtual loss instead of values, so an uncapped
        // loop under a slow evaluator spreads the visits over almost every move.
        while (sims_dispatched < num_simulations &&
               static_cast<int>(in_flight.size()) < max_inflight_batches &&
               static_cast<int>(current_states.size()) < batch_size)
        {
             GomokuState state = root_state;
             std::vector<MctsNode *> path;
             path.reserve(64);
             path.push_back(root_);
             MctsNode *leaf = root_;

             // Traverse
             while (leaf->is_expanded && !leaf->moves.empty()) {
                 leaf = Select(leaf, state);
                 path.push_back(leaf);
             }

             // Check Duplicate
             bool duplicate = false;
             for(auto* l : current_leaves) { if(l == leaf) { duplicate=true; break; } }
             if(!duplicate) {
                 for(const auto& kb : in_flight) {
                     for(auto* l : kb.second.leaves) { if(l == leaf) { duplicate=true; break; } }
                     if(duplicate) break;
                 }
             }

             if (duplicate) break; // Blocked

             // Check Terminal
             int last_move = state.last_move_idx;
             bool win_found = false;
             int winner = 0;
             if (last_move >= 0) {
				int x = last_move % board_size_;
				int y = last_move / board_size_;
				if (core_->check_win(state, x, y)) {
					win_found = true;
					winner = state.board[last_move];
				}
             }
			 if (!win_found && state.empty_count <= 0) winner = 3;

             if (win_found || winner == 3) {
				float value = 0.0f;
				if (win_found) value = (winner == leaf->player_to_play) ? 1.0f : -1.0f;
				Backup(path, value);
				sims_completed++;
                sims_dispatched++;
                made_progress_selection = true;
                continue;
             }

             // Add to Batch
             for(auto* n : path) n->virtual_loss++;
             current_leaves.push_back(leaf);
             current_paths.push_back(std::move(path));
             current_states.push_back(state);
             sims_dispatched++;
             made_progress_selection = true;
        }

        // 2. Dispatch Logic
        if (!current_states.empty()) {
            bool should_dispatch = (static_cast<int>(current_states.size()) >= batch_size)
                                   || (sims_dispatched >= num_simulations) // No more to select
                                   || (!made_progress_selection); // Selection blocked

            if (should_dispatch) {
                AsyncHandle h = dispatch_func(current_states);
                InFlightBatch ifb;
                ifb.handle = h;
                ifb.leaves = std::move(current_leaves);
                ifb.paths = std::move(current_paths);
                ifb.states = std::move(current_states);

                in_flight.insert({h, std::move(ifb)});

                // Clear & re-init
                current_leaves.clear(); current_paths.clear(); current_states.clear();
                current_leaves.reserve(batch_size);
                current_paths.reserve(batch_size);
                current_states.reserve(batch_size);
            }
        }

        // 3. Polling Logic
        if (!in_flight.empty()) {
             std::vector<AsyncHandle> pending_handles;
             pending_handles.reserve(in_flight.size());
             for(const auto& kv : in_flight) pending_handles.push_back(kv.first);

             // CheckAny
             float timeout_s = !made_progress_selection ? 0.002f : 0.0f;
             auto results = check_func(pending_handles, timeout_s);

             for(const auto& res_pair : results) {
                 AsyncHandle h = res_pair.first;
                 const auto& batch_evals = res_pair.second;

                 auto it = in_flight.find(h);
                 if (it == in_flight.end()) continue;

                 InFlightBatch& batch = it->second;

                 if (batch_evals.size() != batch.leaves.size()) throw std::runtime_error("Result size mismatch");

                 for(size_t i=0; i<batch_evals.size(); ++i) {
                     MctsNode* leaf = batch.leaves[i];
                     auto& path = batch.paths[i];
                     const auto& eval = batch_evals[i];

                     for(auto* n : path) n->virtual_loss--;

                     ExpandAtLeaf(leaf, batch.states[i], eval.first);
                     Backup(path, eval.second);
                     sims_completed++;
                 }
                 in_flight.erase(it);
             }

             // No manual sleep needed because check_func handles wait.
        }
    }

	return ChildVisits();
}

MctsNode *MctsEngine::Select(MctsNode *node, GomokuState &state)
{
	if (node == nullptr) {
		throw std::invalid_argument("node is null");
	}
	if (node->moves.empty()) {
		return node;
	}

	bool found = false;
	std::size_t best_index = 0;
	float best_score = -std::numeric_limits<float>::infinity();
	const bool has_children = !node->children.empty();
	for (std::size_t i = 0; i < node->moves.size(); ++i) {
		const float score = CalculateUcb(node, has_children ? node->children[i] : nullptr, node->priors[i]);
		if (score > best_score) {
			best_score = score;
			best_index = i;
			found = true;
		}
	}

	if (!found) {
		throw std::runtime_error("Select failed to find a child");
	}

	if (!has_children) {
		node->children.assign(node->moves.size(), nullptr);
	}
	MctsNode *best_child = node->children[best_index];
	if (best_child == nullptr) {
		best_child = new MctsNode();
		best_child->move_idx = static_cast<int>(node->moves[best_index]);
		best_child->prior = node->priors[best_index];
		best_child->parent = node;
		best_child->player_to_play = 3 - node->player_to_play;
		node->children[best_index] = best_child;
		++node_count_;
	}

	// Apply Move via Core!
	// x, y
	int move = best_child->move_idx;
	int x = move % board_size_;
	int y = move / board_size_;
	// The node's player_to_play is WHO PLAYS at this node.
	// So we apply move for node->player_to_play.
	// state is updated in-place? No, apply_move returns value.
	// GomokuState is simplified struct but apply_move calls C++ internal.
	// GomokuCore::apply_move(const GomokuState &state, ...) returns GomokuState.
	// This is slightly inefficient (copying state), but robust.
	state = core_->apply_move(state, x, y, node->player_to_play);

	return best_child;
}

void MctsEngine::ExpandAtLeaf(
	MctsNode *leaf,
	const GomokuState &state,
	const std::vector<float> &policy
)
{
	if (leaf == nullptr) {
		throw std::invalid_argument("leaf is null");
	}

	const std::vector<int16_t> candidates = core_->get_legal_moves(state);
	if (candidates.empty()) {
		leaf->is_expanded = true;
		return;
	}

	std::vector<std::pair<int, float>> moves;
	moves.reserve(candidates.size());

	for (int16_t idx : candidates) {
		if (idx < 0 || idx >= static_cast<int>(policy.size())) continue;
		const float p = policy.at(static_cast<std::size_t>(idx));
		moves.emplace_back(static_cast<int>(idx), p);
	}

	// Softmax normalization: policy values are raw logits, which can be
	// negative.  Simple sum-division would invert priorities when the sum
	// is negative.  Softmax guarantees a valid probability distribution.
	float max_val = -std::numeric_limits<float>::infinity();
	for (const auto &mv : moves) {
		if (mv.second > max_val) max_val = mv.second;
	}

	float sum = 0.0f;
	for (auto &mv : moves) {
		mv.second = std::exp(mv.second - max_val);
		sum += mv.second;
	}

	if (sum > 0.0f) {
		for (auto &mv : moves) {
			mv.second /= sum;
		}
	} else {
		const float uniform = 1.0f / static_cast<float>(moves.size());
		for (auto &mv : moves) {
			mv.second = uniform;
		}
	}

	leaf->moves.reserve(moves.size());
	leaf->priors.reserve(moves.size());
	for (const auto &mv : moves) {
		leaf->moves.push_back(static_cast<int16_t>(mv.first));
		leaf->priors.push_back(mv.second);
	}
	leaf->is_expanded = true;
}

void MctsEngine::Backup(std::vector<MctsNode *> &path, float leaf_value)
{
	float value = leaf_value;
	for (auto it = path.rbegin(); it != path.rend(); ++it) {
		MctsNode *node = *it;
		if (node == nullptr) {
			continue;
		}
		node->visit_count += 1;
		node->value_sum += value;
		value = -value;
	}
}

float MctsEngine::CalculateUcb(const MctsNode *parent, const MctsNode *child, float prior) const
{
	if (parent == nullptr) {
		throw std::invalid_argument("parent is null");
	}
	const int child_visit_count = child != nullptr ? child->visit_count : 0;
	const int child_virtual_loss = child != nullptr ? child->virtual_loss : 0;
	const int parent_visits = parent->visit_count + parent->virtual_loss;
	const int child_visits = child_visit_count + child_virtual_loss;
	float q = 0.0f;
	if (child_visit_count > 0) {
		// Pending evaluations count as losses so batched selection spreads out.
		q = -(child->value_sum + static_cast<float>(child_virtual_loss))
			/ static_cast<float>(child_visits);
	} else if (fpu_from_parent_ && parent->visit_count > 0) {
		q = parent->value_sum / static_cast<float>(parent->visit_count) - fpu_reduction_;
	}
	const float u = c_puct_ * prior
		* std::sqrt(static_cast<float>(parent_visits + 1))
		/ (1.0f + static_cast<float>(child_visits));
	return q + u;
}
