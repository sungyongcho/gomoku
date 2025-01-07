import { useDoubleThreeLogic } from "./games/useDoubleThreeLogic";
import { useCaptureLogic } from "./games/useCaptureLogic";
import { useWinLogic } from "./games/useWinLogic";

export const useGameLogic = () => {
  const { checkDoubleThree } = useDoubleThreeLogic();
  const { getCapturedStones } = useCaptureLogic();
  const { checkWinCondition } = useWinLogic();

  return {
    getCapturedStones,
    checkDoubleThree,
    checkWinCondition,
  };
};
