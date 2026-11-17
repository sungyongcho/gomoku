<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Evaluation Tool</h1>
      <p class="hint">Right-click (or tap) the faded stone to evaluate that point. The scores are from one evaluation in a real game.</p>

      <div class="stage">
        <div class="board-col">
          <RuleBoard
            :cols="10"
            :rows="8"
            :stones="STONES"
            last="6,1"
            :ghost="{ c: POINT.c, r: POINT.r, v: 1 }"
            :hits="HITS"
            label="Board with one point to evaluate"
            @point="onPoint"
          />
          <div v-if="open" class="tooltip" role="status">
            <div v-for="s in SCORES" :key="s.player" class="row">
              <i :class="['pi', s.player === 'X' ? 'pi-circle-fill' : 'pi-circle']" />
              <EvalScoreGage :model-value="s.percentage" />
            </div>
          </div>
        </div>

        <div class="side">
          <p class="text" v-if="!open">
            Black is to move. Right-click the faded point to ask the minimax engine how good it is for each player.
          </p>
          <template v-else>
            <p class="text">
              <strong>White ○: good.</strong> White already has three on the diagonal running down to this point.
              A white stone here would make four.
            </p>
            <p class="text">
              <strong>Black ●: bad.</strong> The engine scores the same point low for Black.
            </p>
          </template>
          <p class="note">{{ note }}</p>
          <button v-if="open" type="button" class="act" @click="open = false; note = NOTE">Hide scores</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

// The position from the old screenshot (1 black, 2 white); the last move is White's at (6, 1).
const STONES = {
  '6,1': 2,
  '6,2': 2, '7,2': 1,
  '5,3': 1, '6,3': 2,
  '4,4': 1, '5,4': 2, '6,4': 1,
  '3,5': 1, '4,5': 2,
};
const POINT = { c: 3, r: 6 };
// Read off the screenshot's gauges: White near the right end, Black near the left.
const SCORES = [
  { player: 'O', percentage: 97 },
  { player: 'X', percentage: 12 },
];
const NOTE = 'Minimax mode only, and desktop only: touch screens have no right-click.';

const HITS = [];
for (let c = 0; c < 10; c++) {
  for (let r = 0; r < 8; r++) if (!STONES[`${c},${r}`]) HITS.push({ c, r });
}

const open = ref(false);
const note = ref(NOTE);
const onPoint = ({ c, r }) => {
  if (c === POINT.c && r === POINT.r) {
    open.value = !open.value;
    note.value = NOTE;
  } else {
    open.value = false;
    note.value = 'In this example only the faded point has scores. In the game, any empty point can be evaluated.';
  }
};
</script>

<style scoped>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  padding: 24px 16px;
  container-type: inline-size;
}
.container { width: 100%; max-width: 820px; margin: 0 auto; }
h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 16px; }

.stage { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 22px; align-items: center; }
.board-col { position: relative; max-width: 440px; width: 100%; justify-self: center; }

.tooltip {
  position: absolute; left: calc(35% + 26px); top: calc(81.25% - 36px);
  display: flex; flex-direction: column; gap: 8px;
  padding: 8px 10px; border-radius: 6px; background: #fff;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
  font-size: 15px; color: #000; white-space: nowrap;
}
.row { display: flex; align-items: center; gap: 8px; }

.side { display: flex; flex-direction: column; gap: 10px; align-items: flex-start; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }
.note { font-size: 13px; line-height: 1.55; color: #64748b; }
.act {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.act:hover { background: #f1f5f9; border-color: #94a3b8; }

@container (max-width: 600px) {
  .stage { grid-template-columns: 1fr; gap: 14px; }
  .tooltip { position: static; margin: 10px auto 0; width: fit-content; font-size: 13px; }
}
</style>
