cmake_minimum_required(VERSION 3.10)
project(renju_cpp)

# C++ 표준
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# compile_commands.json 생성 (VSCode 인텔리센스용)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- pybind11 경로를 먼저 Python으로부터 가져오기 ---
#   pip install pybind11 한 경우, 이 cmakedir 를 CMAKE_PREFIX_PATH에 추가
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    RESULT_VARIABLE _pybind11_status
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(_pybind11_status EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmakedir}")
endif()

# 이제야 pybind11 찾기 시도
find_package(pybind11 CONFIG REQUIRED)

# --- 순수 C++ 라이브러리 (금수 탐색 로직) ---
add_library(renju_core STATIC
    src/ForbiddenPointFinder.cpp        # 파일 위치에 맞게 조정: src/.. or 그냥 ForbiddenPointFinder.cpp
    src/GomokuCore.cpp
    src/MctsEngine.cpp
)

target_include_directories(renju_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- pybind11 모듈 ---
#   파이썬에서 import 할 이름 = renju_cpp (PYBIND11_MODULE과 일치)
pybind11_add_module(renju_cpp
    src/bindings.cpp                    # 마찬가지로 경로에 맞게 수정
)

# C++ 코어 라이브러리를 파이썬 모듈에 링크
target_link_libraries(renju_cpp
    PRIVATE renju_core
)

# (선택) 헤더 경로가 필요하다면 모듈에도 include 경로 추가
target_include_directories(renju_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# (선택) .so 파일을 특정 파이썬 디렉토리에 떨구고 싶을 때
# 예: 프로젝트 루트의 gomoku 패키지 바로 아래에 두기
set_target_properties(renju_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../gomoku/cpp_ext"
)

# --- pybind11 모듈 (GomokuCore/MCTS Bridge) ---
pybind11_add_module(gomoku_cpp
    src/gomoku_bindings.cpp
)

target_link_libraries(gomoku_cpp
    PRIVATE renju_core
)

target_include_directories(gomoku_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(gomoku_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../gomoku/cpp_ext"
)
