import { useDoubleThreeLogic } from "./games/useDoubleThreeLogic";
import { useCaptureLogic } from "./games/useCaptureLogic";
import { useEndLogic } from "./games/useEndLogic";
import { useBaseLogic } from "./games/useBaseLogic";

export const useGameLogic = () => {
  const { checkDoubleThree } = useDoubleThreeLogic();
  const { getCapturedStones } = useCaptureLogic();
  const { isCaptureEnded, isDrawEnded, isPerfectFiveEnded } = useEndLogic();
  const { getOppositeStone } = useBaseLogic();
  return {
    getOppositeStone,
    getCapturedStones,
    checkDoubleThree,
    isCaptureEnded,
    isDrawEnded,
    isPerfectFiveEnded,
  };
};
