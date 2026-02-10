#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <new>

// MemoryArena: single-buffer allocator to avoid per-node new/delete overhead.
class MemoryArena {
public:
	explicit MemoryArena(std::size_t capacity_bytes)
		: buffer_(capacity_bytes), offset_(0) {
		if (capacity_bytes == 0) {
			throw std::invalid_argument("MemoryArena capacity must be positive");
		}
	}

	MemoryArena(const MemoryArena &) = delete;
	MemoryArena &operator=(const MemoryArena &) = delete;

	MemoryArena(MemoryArena &&) = default;
	MemoryArena &operator=(MemoryArena &&) = default;

	template <typename T>
	T *allocate(std::size_t count = 1, std::size_t align = alignof(T)) {
		static_assert(!std::is_const<T>::value, "Cannot allocate const objects");
		const std::size_t bytes = count * sizeof(T);
		if (align == 0 || (align & (align - 1)) != 0) {
			throw std::invalid_argument("Alignment must be a non-zero power of two");
		}

		std::size_t aligned = (offset_ + (align - 1)) & ~(align - 1);
		if (aligned + bytes > buffer_.size()) {
			throw std::bad_alloc();
		}

		unsigned char *ptr = buffer_.data() + aligned;
		offset_ = aligned + bytes;
		return reinterpret_cast<T *>(ptr);
	}

	void reset() noexcept { offset_ = 0; }

	std::size_t capacity() const noexcept { return buffer_.size(); }
	std::size_t bytes_remaining() const noexcept { return buffer_.size() - offset_; }

private:
	std::vector<unsigned char> buffer_;
	std::size_t offset_;
};

struct MctsNode {
	std::atomic<int> visit_count{0};
	std::atomic<float> value_sum{0.0f};
	std::atomic<int> virtual_loss{0};

float prior{0.0f};
int action{-1};
std::int64_t state_id{-1};  // external handle for Python-side state mapping
MctsNode *parent{nullptr};
MctsNode *children{nullptr};
int num_children{0};
bool is_expanded{false};

MctsNode() = default;

MctsNode(float p, int act, MctsNode *par, std::int64_t sid = -1)
	: prior(p), action(act), state_id(sid), parent(par) {}

private:
	// fetch_add for float is only available in C++20; emulate for C++14.
	static inline float atomic_fetch_add(std::atomic<float> &obj, float arg) {
		float expected = obj.load(std::memory_order_relaxed);
		float desired;
		do {
			desired = expected + arg;
		} while (!obj.compare_exchange_weak(
			expected,
			desired,
			std::memory_order_relaxed,
			std::memory_order_relaxed
		));
		return expected;
	}

public:

	void apply_virtual_loss(int amount = 1) {
		virtual_loss.fetch_add(amount, std::memory_order_relaxed);
	}

	void revert_virtual_loss(int amount = 1) {
		virtual_loss.fetch_sub(amount, std::memory_order_relaxed);
	}

	void backup(float value) {
		atomic_fetch_add(value_sum, value);
		visit_count.fetch_add(1, std::memory_order_relaxed);
		if (virtual_loss.load(std::memory_order_relaxed) > 0) {
			virtual_loss.fetch_sub(1, std::memory_order_relaxed);
		}
	}

	float ucb_score(
		int parent_visits,
		float c_puct,
		float parent_value,
		float fpu_reduction
	) const {
		const int v = visit_count.load(std::memory_order_relaxed);
		float q = 0.0f;
		if (v > 0) {
			q = value_sum.load(std::memory_order_relaxed) / static_cast<float>(v);
		} else {
			q = parent_value * fpu_reduction;
		}
		const float u = c_puct * prior
			* std::sqrt(static_cast<float>(parent_visits + 1))
			/ (1.0f + static_cast<float>(v));
		return q + u - static_cast<float>(virtual_loss.load(std::memory_order_relaxed));
	}
};

struct MctsParams {
	float c_puct{1.0f};
	float fpu_reduction{0.0f};
};

inline MctsNode *expand_node(
	MctsNode &parent,
	MemoryArena &arena,
	const float *priors,
	int num_actions,
	const std::int64_t *state_ids = nullptr,
	const int *action_ids = nullptr
) {
	if (num_actions <= 0) {
		throw std::invalid_argument("num_actions must be positive");
	}
	if (priors == nullptr) {
		throw std::invalid_argument("priors pointer is null");
	}
	parent.children = arena.allocate<MctsNode>(static_cast<std::size_t>(num_actions), alignof(MctsNode));
	parent.num_children = num_actions;
	for (int i = 0; i < num_actions; ++i) {
		const std::int64_t sid = state_ids ? state_ids[i] : static_cast<std::int64_t>(-1);
		const int act = action_ids ? action_ids[i] : i;
		new (&parent.children[i]) MctsNode(priors[i], act, &parent, sid);
	}
	parent.is_expanded = true;
	return parent.children;
}

inline MctsNode *select_child(MctsNode &parent, const MctsParams &params) {
	if (!parent.children || parent.num_children <= 0) {
		return nullptr;
	}
	const int parent_visits = parent.visit_count.load(std::memory_order_relaxed)
		+ parent.virtual_loss.load(std::memory_order_relaxed);
	float parent_value = 0.0f;
	const int pv = parent.visit_count.load(std::memory_order_relaxed);
	if (pv > 0) {
		parent_value = parent.value_sum.load(std::memory_order_relaxed) / static_cast<float>(pv);
	}

	MctsNode *best = nullptr;
	float best_score = -std::numeric_limits<float>::infinity();
	for (int i = 0; i < parent.num_children; ++i) {
		MctsNode &child = parent.children[i];
		const float score = child.ucb_score(parent_visits, params.c_puct, parent_value, params.fpu_reduction);
		if (score > best_score) {
			best_score = score;
			best = &child;
		}
	}
	return best;
}

inline void apply_virtual_loss_path(std::vector<MctsNode *> &path, int amount = 1) {
	for (auto *n : path) {
		if (n != nullptr) {
			n->apply_virtual_loss(amount);
		}
	}
}

inline void revert_virtual_loss_path(std::vector<MctsNode *> &path, int amount = 1) {
	for (auto *n : path) {
		if (n != nullptr) {
			n->revert_virtual_loss(amount);
		}
	}
}

inline void backup_to_root(MctsNode *leaf, float value) {
	MctsNode *cur = leaf;
	float v = value;
	while (cur != nullptr) {
		cur->backup(v);
		v = -v;  // 부모 관점 반전
		cur = cur->parent;
	}
}

// Minimal engine wrapper to drive selection/backup using arena-managed nodes.
class MctsEngine {
public:
	MctsEngine(std::size_t arena_bytes, const MctsParams &params)
		: arena_(arena_bytes), params_(params) {}

	void reset_arena() { arena_.reset(); }

	MctsNode *create_root(const std::vector<float> &priors) {
		arena_.reset();
		MctsNode *root = arena_.allocate<MctsNode>(1, alignof(MctsNode));
		new (root) MctsNode();
		expand_node(*root, arena_, priors.data(), static_cast<int>(priors.size()));
		return root;
	}

	MctsNode *create_root_with_handles(
		const std::vector<float> &priors,
		const std::vector<std::int64_t> *state_ids,
		const std::vector<int> *action_ids
	) {
		arena_.reset();
		MctsNode *root = arena_.allocate<MctsNode>(1, alignof(MctsNode));
		new (root) MctsNode();
		const std::int64_t *sid_ptr = (state_ids && !state_ids->empty()) ? state_ids->data() : nullptr;
		const int *act_ptr = (action_ids && !action_ids->empty()) ? action_ids->data() : nullptr;
		expand_node(
			*root,
			arena_,
			priors.data(),
			static_cast<int>(priors.size()),
			sid_ptr,
			act_ptr
		);
		return root;
	}

	MctsNode *select_best(MctsNode &parent) {
		return select_child(parent, params_);
	}

	void backup(MctsNode *leaf, float value) {
		if (leaf == nullptr) {
			return;
		}
		backup_to_root(leaf, value);
	}

	MemoryArena &arena() { return arena_; }
	MctsParams &params() { return params_; }
	const MemoryArena &arena() const { return arena_; }
	const MctsParams &params() const { return params_; }

private:
	MemoryArena arena_;
	MctsParams params_;
};
