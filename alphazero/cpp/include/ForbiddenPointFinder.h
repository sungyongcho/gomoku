#ifndef FORBIDDEN_POINT_FINDER_H
#define FORBIDDEN_POINT_FINDER_H

#include <vector>

// 기본 보드 크기
static constexpr int DEFAULT_BOARD_SIZE = 19;
#define EMPTYSTONE 0
#define BLACKSTONE 1
#define WHITESTONE 2

// 결과 코드
#define BLACKFIVE 1
#define WHITEFIVE 2
#define BLACKFORBIDDEN 3

// CPoint 구조체 대체
struct CPoint {
    int x, y;
    CPoint(int _x = 0, int _y = 0) : x(_x), y(_y) {}
    bool operator==(const CPoint& other) const { return x == other.x && y == other.y; }
};

class CForbiddenPointFinder {
public:
    explicit CForbiddenPointFinder(int boardSize = DEFAULT_BOARD_SIZE);
    virtual ~CForbiddenPointFinder();

    void Clear();
    void ResizeBoard(int boardSize);
    int AddStone(int x, int y, char cStone);
    void SetStone(int x, int y, char cStone);
    void FindForbiddenPoints();

    // 알고리즘 메서드들
    bool IsFive(int x, int y, int nColor);
    bool IsOverline(int x, int y);
    bool IsFive(int x, int y, int nColor, int nDir);
    bool IsFour(int x, int y, int nColor, int nDir);
    int IsOpenFour(int x, int y, int nColor, int nDir);
    bool IsDoubleFour(int x, int y);
    bool IsOpenThree(int x, int y, int nColor, int nDir);
    bool IsDoubleThree(int x, int y);

    // 멤버 변수 (외부 접근을 위해 public으로 둡니다. 실제 엔지니어링에서는 getter/setter 권장)
    int nForbiddenPoints;
    int boardSize_;
    std::vector<std::vector<char>> cBoard;
    std::vector<CPoint> ptForbidden;
};

#endif
