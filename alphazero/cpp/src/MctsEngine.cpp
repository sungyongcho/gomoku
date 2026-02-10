#include "MctsEngine.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <map>

namespace {
constexpr int kWinLength = 5;
}  // namespace

MctsNode::~MctsNode()
{
	for (auto *child : children) {
		delete child;
	}
}

MctsEngine::MctsEngine(GomokuCore *core, float c_puct)
	: core_(core)
	, board_size_(core != nullptr ? core->board_size() : 0)
	, action_size_(core != nullptr ? core->action_size() : 0)
	, c_puct_(c_puct)
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
	delete root_;
	root_ = nullptr;
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

	delete root_;
	root_ = new MctsNode();
	root_->player_to_play = root_state.next_player;

	for (int i = 0; i < num_simulations; ++i) {
		GomokuState state = root_state; // Copy
		std::vector<MctsNode *> path;
		path.reserve(64);
		path.push_back(root_);
		MctsNode *leaf = root_;

		while (leaf->is_expanded && !leaf->children.empty()) {
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

	std::vector<std::pair<int, int>> visits;
	visits.reserve(root_->children.size());
	for (const auto *child : root_->children) {
		visits.emplace_back(child->move_idx, child->visit_count);
	}
	return visits;
}

std::vector<std::pair<int, int>> MctsEngine::RunSimulationEncoded(
	const GomokuState &root_state,
	int num_simulations,
	EvaluationCallbackEncoded eval_func
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

	delete root_;
	root_ = new MctsNode();
	root_->player_to_play = root_state.next_player;

	for (int i = 0; i < num_simulations; ++i) {
		GomokuState state = root_state;
		std::vector<MctsNode *> path;
		path.reserve(64);
		path.push_back(root_);
		MctsNode *leaf = root_;

		while (leaf->is_expanded && !leaf->children.empty()) {
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
	}

	std::vector<std::pair<int, int>> visits;
	visits.reserve(root_->children.size());
	for (const auto *child : root_->children) {
		visits.emplace_back(child->move_idx, child->visit_count);
	}
	return visits;
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

	delete root_;
	root_ = new MctsNode();
	root_->player_to_play = root_state.next_player;

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
			while (leaf->is_expanded && !leaf->children.empty()) {
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

	std::vector<std::pair<int, int>> visits;
	visits.reserve(root_->children.size());
	for (const auto *child : root_->children) {
		visits.emplace_back(child->move_idx, child->visit_count);
	}
	return visits;
}

std::vector<std::pair<int, int>> MctsEngine::RunAsyncSimulation(
	const GomokuState &root_state,
	int num_simulations,
	int batch_size,
	AsyncDispatchCallback dispatch_func,
	AsyncCheckCallback check_func
)
{
	if (static_cast<int>(root_state.board.size()) != action_size_) {
		throw std::invalid_argument("root_state size mismatch");
	}
	if (num_simulations <= 0) return {};
	if (!dispatch_func || !check_func) {
		throw std::invalid_argument("Async callbacks must be provided");
	}
	if (batch_size <= 0) {
		throw std::invalid_argument("batch_size must be positive");
	}

	delete root_;
	root_ = new MctsNode();
	root_->player_to_play = root_state.next_player;

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

        // 1. Selection Phase
        while (sims_dispatched < num_simulations &&
               static_cast<int>(current_states.size()) < batch_size)
        {
             GomokuState state = root_state;
             std::vector<MctsNode *> path;
             path.reserve(64);
             path.push_back(root_);
             MctsNode *leaf = root_;

             // Traverse
             while (leaf->is_expanded && !leaf->children.empty()) {
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

	std::vector<std::pair<int, int>> visits;
	visits.reserve(root_->children.size());
	for (const auto *child : root_->children) {
		visits.emplace_back(child->move_idx, child->visit_count);
	}
	return visits;
}

MctsNode *MctsEngine::Select(MctsNode *node, GomokuState &state)
{
	if (node == nullptr) {
		throw std::invalid_argument("node is null");
	}
	if (node->children.empty()) {
		return node;
	}

	MctsNode *best_child = nullptr;
	float best_score = -std::numeric_limits<float>::infinity();
	for (auto *child : node->children) {
		const float score = CalculateUcb(node, child);
		if (score > best_score) {
			best_score = score;
			best_child = child;
		}
	}

	if (best_child == nullptr) {
		throw std::runtime_error("Select failed to find a child");
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

	for (const auto &mv : moves) {
		auto *child = new MctsNode();
		child->move_idx = mv.first;
		child->prior = mv.second;
		child->parent = leaf;
		child->player_to_play = 3 - leaf->player_to_play;
		leaf->children.push_back(child);
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

float MctsEngine::CalculateUcb(const MctsNode *parent, const MctsNode *child) const
{
	if (parent == nullptr || child == nullptr) {
		throw std::invalid_argument("parent or child is null");
	}
	const int parent_visits = parent->visit_count + parent->virtual_loss;
	const int child_visits = child->visit_count + child->virtual_loss;
	float q = 0.0f;
	if (child_visits > 0) {
		q = -(child->value_sum / static_cast<float>(child_visits));
	}
	const float u = c_puct_ * child->prior
		* std::sqrt(static_cast<float>(parent_visits + 1))
		/ (1.0f + static_cast<float>(child_visits));
	return q + u;
}
