import logging

# ANSI 색상 코드
COLORS = {
    "RaySelfPlayWorker": "\033[34m",  # 파란색
    "RayInferenceServer": "\033[32m",  # 초록색
}
RESET = "\033[0m"


class ColoredMessageFormatter(logging.Formatter):
    """
    로그 메시지 본문에만 색상을 적용하는 커스텀 포맷터.
    헤더(접두사)는 Ray가 자동으로 붙이도록 둡니다.
    """

    def __init__(self, actor_name: str):
        super().__init__()
        self.color = COLORS.get(actor_name, "")

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        # 메시지 본문에만 색상 적용
        return f"{self.color}{message}{RESET}"


def setup_actor_logging(actor_name: str) -> logging.Logger:
    """
    Ray 환경에 맞는 간단한 커스텀 로거를 설정합니다.
    """
    logger = logging.getLogger(actor_name)
    # logger.propagate = False를 주석 처리하거나 제거하여 Ray의 기본 핸들러와 충돌하지 않도록 합니다.
    # logger.propagate = False
    logger.setLevel(logging.INFO)

    # 핸들러가 중복 추가되는 것을 방지합니다.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = ColoredMessageFormatter(actor_name)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
