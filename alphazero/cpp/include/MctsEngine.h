#pragma once

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
		EvaluationCallbackEncoded eval_func,
		float time_limit_s = 0.0f
	);

	// (move_idx, prior) of the root children from the last search.
	std::vector<std::pair<int, float>> RootPriors() const;

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
		AsyncCheckCallback check_func
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

	MctsNode *Select(MctsNode *node, GomokuState &state);
	void ExpandAtLeaf(
		MctsNode *leaf,
		const GomokuState &state,
		const std::vector<float> &policy
	);
	void Backup(std::vector<MctsNode *> &path, float leaf_value);
	float CalculateUcb(const MctsNode *parent, const MctsNode *child) const;
};
