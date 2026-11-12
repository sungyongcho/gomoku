#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "GomokuCore.h"

struct MctsNode {
	int move_idx{-1};
	int player_to_play{1};
	int visit_count{0};
	int virtual_loss{0};
	float value_sum{0.0f};
	float prior{0.0f};
	bool is_expanded{false};
	// Legal moves of an expanded node with their priors. children is empty until the search
	// first goes below this node; from then on it runs parallel to moves, with an entry
	// staying null until that move is first selected. Memory and the cost of freeing a tree
	// therefore follow the number of searches, not the number of legal moves.
	std::vector<int16_t> moves;
	std::vector<float> priors;
	std::vector<MctsNode *> children;
	MctsNode *parent{nullptr};

	~MctsNode();
};

using EvaluationResult = std::pair<std::vector<float>, float>;
using EvaluationCallback = std::function<EvaluationResult(const GomokuState &)>;
using EvaluationCallbackEncoded = std::function<
	EvaluationResult(const GomokuState &, int player_to_play, int last_move_idx)
>;
using EvaluationCallbackBatch = std::function<
	std::vector<EvaluationResult>(std::vector<GomokuState>&)
>;

using AsyncHandle = int;
using AsyncDispatchCallback = std::function<AsyncHandle(std::vector<GomokuState>&)>;
// CheckAny: Takes list of pending handles and timeout_s. Returns list of (Handle, Result) for those ready.
using AsyncCheckCallback = std::function<
	std::vector<std::pair<AsyncHandle, std::vector<EvaluationResult>>>(const std::vector<AsyncHandle>&, float)
>;

class MctsEngine {
public:
	// fpu_from_parent scores unvisited moves with the parent's mean value instead of 0,
	// lowered by fpu_reduction so that search deepens on moves it has already tried.
	MctsEngine(GomokuCore *core, float c_puct = 5.0f, bool fpu_from_parent = false, float fpu_reduction = 0.0f);
	~MctsEngine();

	std::vector<std::pair<int, int>> RunSimulation(
		const GomokuState &root_state,
		int num_simulations,
		EvaluationCallback eval_func
	);

	// time_limit_s > 0 stops the search once the budget is spent, after at least one simulation.
	std::vector<std::pair<int, int>> RunSimulationEncoded(
		const GomokuState &root_state,
		int num_simulations,
		const EvaluationCallbackEncoded &eval_func,
		float time_limit_s = 0.0f
	);

	// (move_idx, prior) of the root children from the last search.
	std::vector<std::pair<int, float>> RootPriors() const;

	// --- Keeping the tree between searches (serving) ---
	// Every Run* call above starts a new tree. The calls below keep one: ResetRoot starts it,
	// ContinueSimulationEncoded adds searches to it, and AdvanceRoot / AlignRoot move the
	// root down as the game goes on, keeping the searches already spent below the move.

	// Drop the tree and start a new one at root_state.
	void ResetRoot(const GomokuState &root_state);
	// Drop the tree without starting a new one.
	void ClearTree();
	bool HasRoot() const { return root_ != nullptr; }
	// Searches that have passed through the root, 0 without a tree.
	int RootVisits() const { return root_ != nullptr ? root_->visit_count : 0; }
	// Nodes in the kept tree.
	int NodeCount() const { return node_count_; }
	// Searches completed by the last ContinueSimulationEncoded call.
	int LastSimulations() const { return last_simulations_; }

	// Make the root's child for move_idx the new root and free the rest of the tree.
	// Returns false, leaving the tree untouched, when move_idx is not a move of the root.
	bool AdvanceRoot(int move_idx);

	// Point the tree at target when it is the root position itself or the position one move
	// below it, compared field by field (board, capture points, side to move, last move,
	// empty count and move history). Returns the searches carried over, or -1 when target is
	// neither, in which case the tree is left untouched and the caller starts a new one.
	int AlignRoot(const GomokuState &target);

	// Add searches to the kept tree. Stops after num_simulations, after time_limit_s (> 0),
	// once the tree holds max_nodes nodes (> 0), or, when stoppable, as soon as RequestStop
	// is called from another thread. A root that has never been searched gets one search
	// before the time limit applies, so that there is always a move to return.
	std::vector<std::pair<int, int>> ContinueSimulationEncoded(
		int num_simulations,
		const EvaluationCallbackEncoded &eval_func,
		float time_limit_s = 0.0f,
		int max_nodes = 0,
		bool stoppable = false
	);

	// The stop flag stays set until ClearStop, so a stop requested before a stoppable search
	// has entered its loop still ends it. Both are safe to call from any thread.
	void RequestStop() { stop_requested_.store(true, std::memory_order_relaxed); }
	void ClearStop() { stop_requested_.store(false, std::memory_order_relaxed); }

	std::vector<std::pair<int, int>> RunBatchSimulation(
		const GomokuState &root_state,
		int num_simulations,
		int batch_size,
		EvaluationCallbackBatch eval_func
	);

	std::vector<std::pair<int, int>> RunAsyncSimulation(
		const GomokuState &root_state,
		int num_simulations,
		int batch_size,
		AsyncDispatchCallback dispatch_func,
		AsyncCheckCallback check_func,
		// Batches allowed in flight at once. 1 makes the search match RunBatchSimulation; larger
		// values overlap inference with selection but spread visits under a slow evaluator.
		int max_inflight_batches = 1
	);

	GomokuCore *core() const { return core_; }

private:
	GomokuCore *core_;
	int board_size_;
	int action_size_;
	float c_puct_;
	bool fpu_from_parent_;
	float fpu_reduction_;
	MctsNode *root_{nullptr};
	GomokuState root_state_{};
	int node_count_{0};
	int last_simulations_{0};
	std::atomic<bool> stop_requested_{false};

	std::vector<std::pair<int, int>> ChildVisits() const;
	void AdvanceToChild(std::size_t index, GomokuState next_state);
	MctsNode *Select(MctsNode *node, GomokuState &state);
	void ExpandAtLeaf(
		MctsNode *leaf,
		const GomokuState &state,
		const std::vector<float> &policy
	);
	void Backup(std::vector<MctsNode *> &path, float leaf_value);
	// child is null for a move that has not been visited yet.
	float CalculateUcb(const MctsNode *parent, const MctsNode *child, float prior) const;
};
