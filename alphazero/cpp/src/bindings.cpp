#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ForbiddenPointFinder.h"

namespace py = pybind11;

// 요청하신 Python 함수 프로토타입:
// detect_doublethree(board: np.ndarray, x: int, y: int, player: int, board_size: int) -> bool
bool detect_doublethree_wrapper(py::array_t<int, py::array::c_style | py::array::forcecast> board_array, int x, int y, int player, int board_size)
{
	// 1. 입력 유효성 검사 (안전 장치)
	if (board_size <= 0)
	{
		throw std::runtime_error("Board size must be positive");
	}

	// 2. Numpy 배열 접근을 위한 버퍼 정보 가져오기
	py::buffer_info buf = board_array.request();
	if (buf.ndim != 2)
	{
		throw std::runtime_error("Number of dimensions must be 2");
	}

	// 3. CForbiddenPointFinder 인스턴스 생성 및 초기화
	const int height = static_cast<int>(buf.shape[0]);
	const int width = static_cast<int>(buf.shape[1]);
	if (height != width)
	{
		throw std::runtime_error("Board must be a square matrix");
	}
	if (height != board_size)
	{
		throw std::runtime_error("board_size argument must match board dimensions");
	}
	if (player != BLACKSTONE && player != WHITESTONE)
	{
		throw std::runtime_error("player must be 1 (black) or 2 (white)");
	}

	CForbiddenPointFinder finder(board_size);

	// 4. Numpy 배열의 데이터를 C++ 내부 cBoard로 복사
	// Python Numpy는 보통 0-based 인덱스이지만, C++ 로직은 내부에 padding을 둡니다.
	// finder.SetStone 메서드가 내부적으로 x+1, y+1 처리를 하므로 0-based x, y를 넘기면 됩니다.

	int *ptr = static_cast<int *>(buf.ptr); // Numpy 데이터 포인터

	for (int y = 0; y < board_size; ++y)
	{
		for (int x = 0; x < board_size; ++x)
		{
			// ptr 접근: row * stride_row + col * stride_col 방식이 안전하지만,
			// 연속된 C-style array라고 가정하고 간단히 처리합니다.
			int stone = ptr[y * width + x];
			if (stone == EMPTYSTONE)
			{
				continue;
			}
			if (stone != BLACKSTONE && stone != WHITESTONE)
			{
				throw std::runtime_error("board cell must be 0, 1, or 2");
			}
			// stone 값 매핑: 입력 numpy가 1=Black, 2=White라고 가정
			// 헤더의 상수: EMPTYSTONE=0, BLACKSTONE=1, WHITESTONE=2
			// Finder는 흑 기준 금수 로직이므로 백 차례에서는 색을 반전해 동일 로직으로 평가한다.
			int mapped_stone = stone;
			if (player == WHITESTONE)
			{
				if (stone == BLACKSTONE)
				{
					mapped_stone = WHITESTONE;
				}
				else if (stone == WHITESTONE)
				{
					mapped_stone = BLACKSTONE;
				}
			}
			finder.SetStone(x, y, static_cast<char>(mapped_stone));
		}
	}


	// IsDoubleThree는 해당 위치(x,y)에 놓았을 때 3-3이 되는지 확인합니다.
	return finder.IsDoubleThree(x, y);
}

// 모듈 정의 (모듈 이름: renju_cpp)
PYBIND11_MODULE(renju_cpp, m)
{
	m.doc() = "Renju forbidden point finder logic backed by C++"; // 모듈 설명

	m.def("detect_doublethree", &detect_doublethree_wrapper,
		  "Detect double three forbidden point",
		  py::arg("board"), py::arg("x"), py::arg("y"), py::arg("player"), py::arg("board_size"));
}
