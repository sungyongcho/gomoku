
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
