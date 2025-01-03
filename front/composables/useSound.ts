import stoneMp3 from "~/assets/stone.mp3";
import undoMp3 from "~/assets/undo.mp3";

export const useSound = () => {
  const playStoneSound = () => {
    const audio = new Audio(stoneMp3);
    audio.play();
  };

  const playUndoSound = () => {
    const audio = new Audio(undoMp3);
    audio.play();
  };

  return {
    playStoneSound,
    playUndoSound,
  };
};
