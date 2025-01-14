import { useDoubleThreeLogic } from "./games/useDoubleThreeLogic";
import { useCaptureLogic } from "./games/useCaptureLogic";
import { useEndLogic } from "./games/useEndLogic";

export const useGameLogic = () => {
  const { checkDoubleThree } = useDoubleThreeLogic();
  const { getCapturedStones } = useCaptureLogic();
  const {
    isCaptureEnded,
    isCurrentTurnFiveEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  } = useEndLogic();

  return {
    getCapturedStones,
    checkDoubleThree,
    isCaptureEnded,
    isCurrentTurnFiveEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  };
};
