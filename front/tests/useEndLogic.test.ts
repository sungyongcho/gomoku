import { describe, expect, it, vi } from "vitest";
import { ref } from "vue";
import { useEndLogic } from "~/composables/games/useEndLogic";
import { GAME_END_SCENARIO, type GameSituation } from "~/types/game";

// isCaptureEnded reads the settings through Nuxt auto-imports.
const settings = ref({ firstMove: "Player1" });
vi.stubGlobal("useGameStore", () => ({}));
vi.stubGlobal("storeToRefs", () => ({ settings }));

const situation = (player1: number, player2: number, goal = 5): GameSituation => ({
  x: 0,
  y: 0,
  boardData: [],
  turn: "X",
  captured: { player1, player2, goal },
});

describe("isCaptureEnded", () => {
  const { isCaptureEnded } = useEndLogic();

  // Black (X) is whichever player moves first.
  it.each([
    { firstMove: "Player1", reachedBy: "Player1", winner: "X" },
    { firstMove: "Player1", reachedBy: "Player2", winner: "O" },
    { firstMove: "Player2", reachedBy: "Player1", winner: "O" },
    { firstMove: "Player2", reachedBy: "Player2", winner: "X" },
  ])(
    "$firstMove moves first, $reachedBy reaches the goal: winner $winner",
    ({ firstMove, reachedBy, winner }) => {
      settings.value.firstMove = firstMove;
      const [player1, player2] = reachedBy === "Player1" ? [5, 0] : [0, 5];
      expect(isCaptureEnded(situation(player1, player2))).toEqual({
        result: GAME_END_SCENARIO.PAIR_CAPTURED,
        winner,
      });
    },
  );

  it("uses the configured capture goal", () => {
    settings.value.firstMove = "Player2";
    expect(isCaptureEnded(situation(3, 0, 3))).toEqual({
      result: GAME_END_SCENARIO.PAIR_CAPTURED,
      winner: "O",
    });
  });

  it("does not end the game below the goal", () => {
    expect(isCaptureEnded(situation(4, 4))).toEqual({});
  });
});
