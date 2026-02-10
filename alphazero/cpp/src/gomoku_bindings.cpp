#include <memory>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "GomokuCore.h"
#include "MctsEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(gomoku_cpp, m)
{
	m.doc() = "Gomoku core rules and encoding implemented in C++ (core-only)";

	py::class_<GomokuState>(m, "GomokuState")
		.def(py::init<>())
		.def_readwrite("board", &GomokuState::board)
		.def_readwrite("p1_pts", &GomokuState::p1_pts)
		.def_readwrite("p2_pts", &GomokuState::p2_pts)
		.def_readwrite("next_player", &GomokuState::next_player)
		.def_readwrite("last_move_idx", &GomokuState::last_move_idx)
		.def_readwrite("empty_count", &GomokuState::empty_count)
		.def_readwrite("history", &GomokuState::history);

	py::class_<GomokuCore>(m, "GomokuCore")
		.def(py::init<int, bool, bool, int, int, int>(),
			py::arg("board_size"),
			py::arg("enable_doublethree"),
			py::arg("enable_capture"),
			py::arg("capture_goal"),
			py::arg("gomoku_goal"),
			py::arg("history_length"))
		.def_property_readonly("board_size", &GomokuCore::board_size)
		.def_property_readonly("action_size", &GomokuCore::action_size)
		.def_property_readonly("history_length", &GomokuCore::history_length)
		.def("initial_state", &GomokuCore::initial_state)
		.def("apply_move", &GomokuCore::apply_move, py::arg("state"), py::arg("x"), py::arg("y"), py::arg("player"))
        .def("check_win", &GomokuCore::check_win, py::arg("state"), py::arg("x"), py::arg("y"))
        .def("get_legal_moves", &GomokuCore::get_legal_moves, py::arg("state"))
        .def(
            "get_candidate_moves",
            &GomokuCore::GetCandidateMoves,
            py::arg("board_state"),
            "Get legal moves within 2 steps from existing stones"
        )
        .def(
            "encode_state",
            [](const GomokuCore &core, const GomokuState &state) {
                auto features = core.encode_state(state);
                const ssize_t b = core.board_size();
                const ssize_t total_channels = static_cast<ssize_t>(8 + core.history_length());
                const ssize_t shape[3] = {total_channels, b, b};
                auto *buffer = new std::vector<float>(std::move(features));
                py::capsule free_when_done(buffer, [](void *ptr) {
                    delete static_cast<std::vector<float> *>(ptr);
                });
                return py::array_t<float>(
                    shape,
                    buffer->data(),
                    free_when_done
                );
            },
            py::arg("state"),
            py::return_value_policy::move
        )
        .def(
            "write_state_features",
            [](const GomokuCore &core, const GomokuState &state) {
                const ssize_t b = core.board_size();
                const ssize_t total_channels = static_cast<ssize_t>(8 + core.history_length());
                const ssize_t plane = b * b;
                const ssize_t shape[3] = {total_channels, b, b};
                const ssize_t strides[3] = {
                    static_cast<ssize_t>(sizeof(float) * plane),
                    static_cast<ssize_t>(sizeof(float) * b),
                    static_cast<ssize_t>(sizeof(float))
                };
                auto *buffer = new std::vector<float>(static_cast<std::size_t>(total_channels * plane));
                core.write_state_features(state, buffer->data());
                py::capsule free_when_done(buffer, [](void *ptr) {
                    delete static_cast<std::vector<float> *>(ptr);
                });
                return py::array(
                    shape,
                    strides,
                    buffer->data(),
                    free_when_done
                );
            },
            py::arg("state"),
            py::return_value_policy::move
        );

	py::class_<MctsEngine>(m, "MctsEngine")
		.def(py::init<GomokuCore *, float>(), py::arg("core"), py::arg("c_puct") = 5.0f)
		.def(
			"run_mcts", // Python에서는 run_mcts로 호출
			[](MctsEngine &engine, const GomokuState &state, int sims, py::function evaluator) {
				// 1. Python Evaluator 래퍼 (C++ -> Python 호출 시 GIL 획득)
				EvaluationCallback callback = [evaluator](const GomokuState &s) {
					py::gil_scoped_acquire gil; // Python 객체 접근 전 Lock
					// GomokuState는 pybind11에 의해 자동으로 변환(또는 복사)되어 전달됨
					py::object result = evaluator(s);

					if (!py::isinstance<py::tuple>(result)) {
						throw std::runtime_error("evaluator must return (policy, value)");
					}
					const py::tuple tup = result.cast<py::tuple>();
					// Python 리스트/배열을 C++ 벡터로 변환
					const auto policy = tup[0].cast<std::vector<float>>();
					const float value = tup[1].cast<float>();
					return EvaluationResult{policy, value};
				};

				// 2. 시뮬레이션 실행 (GIL 해제 상태로 고속 연산)
				py::gil_scoped_release release; // C++ 루프 도는 동안 Python 다른 스레드 허용
				return engine.RunSimulation(state, sims, callback);
			},
			py::arg("state"),
			py::arg("sims"),
			py::arg("evaluator"),
			"Run MCTS simulation."
		)
		.def(
			"run_mcts_encoded",
			[](MctsEngine &engine, const GomokuState &state, int sims, py::function evaluator) {
				// 1. Python Evaluator Wrapper for Encoded State
				EvaluationCallbackEncoded callback = [&engine, evaluator](const GomokuState &s, int player, int last_move) {
					// NOTE: This runs inside the MCTS loop (thread). We need GIL to call Python.
					py::gil_scoped_acquire gil;

					// 2. Encode features directly in C++ using the FULL GomokuState `s`
					GomokuCore* core = engine.core();
					const int b = core->board_size();
					const int total_channels = 8 + core->history_length();
					const int plane = b * b;

					auto *feature_buf = new std::vector<float>(static_cast<std::size_t>(total_channels * plane));
					core->write_state_features(s, feature_buf->data());

					// Create Numpy array for features (transfer ownership to Python caps)
					py::capsule free_feat(feature_buf, [](void *ptr) {
						delete static_cast<std::vector<float> *>(ptr);
					});
					const ssize_t f_shape[3] = {static_cast<ssize_t>(total_channels), b, b};
					auto py_features = py::array_t<float>(f_shape, feature_buf->data(), free_feat);

					// 3. Get Legal Moves directly in C++
					std::vector<int16_t> legal_moves = core->get_legal_moves(s);

					// Create Numpy array for legal indices
					auto py_legal = py::array(py::cast(legal_moves));

					// 4. Call Python Evaluator: (features, legal_indices) -> (policy, value)
					py::object result = evaluator(py_features, py_legal);

					if (!py::isinstance<py::tuple>(result)) {
						throw std::runtime_error("evaluator must return (policy, value)");
					}
					const py::tuple tup = result.cast<py::tuple>();
					const auto policy = tup[0].cast<std::vector<float>>();
					const float value = tup[1].cast<float>();

					return EvaluationResult{policy, value};
				};

				py::gil_scoped_release release;
				return engine.RunSimulationEncoded(state, sims, callback);
			},
			py::arg("state"),
			py::arg("sims"),
			py::arg("evaluator"),
			"Run MCTS simulation with native feature encoding and legal move generation."
			"Run MCTS simulation with native feature encoding and legal move generation."
		)
		.def(
			"run_mcts_batch_encoded",
			[](MctsEngine &engine, const GomokuState &state, int sims, int batch_size, py::function evaluator) {
				// 1. Python Evaluator Callback for Batch
				EvaluationCallbackBatch callback = [&engine, evaluator](std::vector<GomokuState> &states) {
					py::gil_scoped_acquire gil;

					if (states.empty()) return std::vector<EvaluationResult>{};

					GomokuCore* core = engine.core();
					const int b = core->board_size();
					const int total_channels = 8 + core->history_length();
					const int plane = b * b;
					const int n = static_cast<int>(states.size());

					// Allocate Batch Tensor Buffer: (N, C, H, W) nchw
					auto *batch_buf = new std::vector<float>(static_cast<std::size_t>(n * total_channels * plane));
					float* ptr = batch_buf->data();

					// Fill buffer
					for (const auto& s : states) {
						core->write_state_features(s, ptr);
						ptr += (total_channels * plane);
					}

					// Wrap in Numpy
					py::capsule free_buf(batch_buf, [](void *ptr) {
						delete static_cast<std::vector<float> *>(ptr);
					});
					const ssize_t shape[4] = {n, total_channels, b, b};
					auto py_batch = py::array_t<float>(shape, batch_buf->data(), free_buf);

					// Call Python: evaluator(tensor) -> (policy_batch, value_batch)
					// Note: We do NOT pass legal moves list here for efficiency.
					// The Python side must handle simple inference.
					py::object result = evaluator(py_batch);

					if (!py::isinstance<py::tuple>(result)) {
						throw std::runtime_error("evaluator must return (policy_batch, value_batch)");
					}
					const py::tuple tup = result.cast<py::tuple>();

					// Check shapes? We expect lists or numpy arrays.
					// We assume numpy arrays for efficiency.
					auto policies = tup[0].cast<py::array_t<float>>(); // (N, ActionSize)
					auto values = tup[1].cast<py::array_t<float>>();   // (N, 1) or (N,)

					py::buffer_info p_info = policies.request();
					py::buffer_info v_info = values.request();

					if (p_info.shape[0] != n) throw std::runtime_error("Policy batch size mismatch");

					float* p_ptr = static_cast<float*>(p_info.ptr);
					float* v_ptr = static_cast<float*>(v_info.ptr);

					const int action_dim = static_cast<int>(p_info.shape[1]);
					if (action_dim != core->action_size()) throw std::runtime_error("Action size mismatch");

					std::vector<EvaluationResult> results;
					results.reserve(n);

					for (int i = 0; i < n; ++i) {
						std::vector<float> pol(p_ptr + i * action_dim, p_ptr + (i + 1) * action_dim);
						float val = v_ptr[i]; // Assuming contiguous (N,1) or (N,)
						results.emplace_back(std::move(pol), val);
					}

					return results;
				};

				py::gil_scoped_release release;
				return engine.RunBatchSimulation(state, sims, batch_size, callback);
			},
			py::arg("state"),
			py::arg("sims"),
			py::arg("batch_size"),
			py::arg("evaluator"),
			"Run MCTS Batch simulation. Evaluator receives (B, C, H, W) tensor and returns (PolicyBatch, ValueBatch)."
		)
        .def(
            "run_mcts_async_encoded",
            [](MctsEngine &engine, const GomokuState &state, int sims, int batch_size, py::function dispatcher, py::function checker) {
                // Dispatch: (vector<State>) -> Handle
                AsyncDispatchCallback dispatch = [&engine, dispatcher](std::vector<GomokuState> &states) -> AsyncHandle {
                    py::gil_scoped_acquire gil;
                    if (states.empty()) return -1;

					GomokuCore* core = engine.core();
					const int b = core->board_size();
					const int total_channels = 8 + core->history_length();
					const int plane = b * b;
					const int n = static_cast<int>(states.size());

					// Allocate Batch Tensor
					auto *batch_buf = new std::vector<float>(static_cast<std::size_t>(n * total_channels * plane));
					float* ptr = batch_buf->data();
					for (const auto& s : states) {
						core->write_state_features(s, ptr);
						ptr += (total_channels * plane);
					}

					// Wrap in Numpy
					py::capsule free_buf(batch_buf, [](void *ptr) { delete static_cast<std::vector<float> *>(ptr); });
					const ssize_t shape[4] = {n, total_channels, b, b};
					auto py_batch = py::array_t<float>(shape, batch_buf->data(), free_buf);

                    // Call dispatcher
                    py::object res = dispatcher(py_batch);
                    return res.cast<int>();
                };

                // Check: (vector<Handle>, float) -> vector<pair<Handle, vector<Result>>>
                AsyncCheckCallback check = [&engine, checker](const std::vector<AsyncHandle>& handles, float timeout_s) {
                    py::gil_scoped_acquire gil;
                    std::vector<std::pair<AsyncHandle, std::vector<EvaluationResult>>> out_results;
                    if (handles.empty()) return out_results;

                    // Call checker
                    py::list py_handles;
                    for(auto h : handles) py_handles.append(h);

                    py::object res_obj = checker(py_handles, timeout_s);
                    py::list res_list = res_obj.cast<py::list>();

                    out_results.reserve(res_list.size());
                    GomokuCore* core = engine.core(); // needed for action check validation if desired
                    (void) core;

                    for(auto item : res_list) {
                        // item is (handle, (p_batch, v_batch))
                        auto pair = item.cast<py::tuple>();
                        AsyncHandle h = pair[0].cast<int>();
                        auto batch_res = pair[1].cast<py::tuple>();

                        auto policies = batch_res[0].cast<py::array_t<float>>();
                        auto values = batch_res[1].cast<py::array_t<float>>();

    					py::buffer_info p_info = policies.request();
    					py::buffer_info v_info = values.request();

                        int n = static_cast<int>(p_info.shape[0]);
                        int action_dim = static_cast<int>(p_info.shape[1]); // Ensure 2D

    					float* p_ptr = static_cast<float*>(p_info.ptr);
    					float* v_ptr = static_cast<float*>(v_info.ptr);

                        std::vector<EvaluationResult> batch_vec;
                        batch_vec.reserve(n);

    					for (int i = 0; i < n; ++i) {
    						std::vector<float> pol(p_ptr + i * action_dim, p_ptr + (i + 1) * action_dim);
    						float val = v_ptr[i];
    						batch_vec.emplace_back(std::move(pol), val);
    					}
                        out_results.push_back({h, std::move(batch_vec)});
                    }
                    return out_results;
                };

                py::gil_scoped_release release;
                return engine.RunAsyncSimulation(state, sims, batch_size, dispatch, check);
            },
			py::arg("state"),
			py::arg("sims"),
			py::arg("batch_size"),
            py::arg("dispatcher"),
            py::arg("checker"),
            "Run Async MCTS."
        );
	}
