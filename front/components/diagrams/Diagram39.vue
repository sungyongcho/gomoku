<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>History Mode</h1>
      <p class="hint">Press <strong>History</strong>, then step through the game with <strong>Prev</strong> and <strong>Next</strong>. The live game is not touched.</p>

      <div class="stage">
        <div class="board-col">
          <RuleBoard
            :cols="WIN_C.length"
            :rows="WIN_R.length"
            :stones="stones"
            :last="lastKey"
            :tone="on ? 'history' : 'wood'"
            :col-labels="WIN_C.map((c) => letter(c))"
            :row-labels="WIN_R.map((r) => String(r + 1))"
            :label="on ? `History mode, move ${shown} of ${MOVES.length}` : 'Live game'"
          />
          <div class="buttons">
            <template v-if="!on">
              <button type="button" class="btn" disabled><i class="pi pi-undo" /> Undo a turn</button>
              <button type="button" class="btn" disabled><i class="pi pi-play" /> Restart</button>
            </template>
            <template v-else>
              <button type="button" class="btn" :disabled="shown === 0" @click="shown--"><i class="pi pi-arrow-left" /> Prev</button>
              <button type="button" class="btn" :disabled="shown === MOVES.length" @click="shown++">Next <i class="pi pi-arrow-right" /></button>
            </template>
          </div>
        </div>

        <div class="side">
          <section class="panel">
            <div class="panel-head">
              <button type="button" :class="['hist', { active: on }]" :aria-pressed="on" @click="toggle">History</button>
              <span class="exports">url <i class="pi pi-link" /> &nbsp; json <i class="pi pi-download" /></span>
            </div>
            <ul class="log">
              <li v-for="(m, i) in MOVES.slice(0, on ? shown : MOVES.length)" :key="i">
                [{{ m.v === 1 ? 'Black' : 'White' }}]: ({{ letter(m.c) }}, {{ m.r + 1 }})
              </li>
            </ul>
          </section>
          <p class="text">
            <template v-if="!on">The live game: 8 moves played. Undo and Restart act on this board.</template>
            <template v-else>History mode: the board turns grey and stops taking moves. It shows move {{ shown }} of {{ MOVES.length }}, and the log lists only the moves up to it.</template>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

// Full-board coordinates (column A = 0, row 1 = 0), the game from the old screenshot.
const MOVES = [
  { c: 9, r: 8, v: 1 }, // J9
  { c: 8, r: 8, v: 2 }, // I9
  { c: 8, r: 7, v: 1 }, // I8
  { c: 10, r: 9, v: 2 }, // K10
  { c: 9, r: 7, v: 1 }, // J8
  { c: 8, r: 6, v: 2 }, // I7
  { c: 7, r: 7, v: 1 }, // H8
  { c: 10, r: 7, v: 2 }, // K8
];
// The diagram shows columns E..M and rows 4..12 of the 19x19 board.
const WIN_C = [4, 5, 6, 7, 8, 9, 10, 11, 12];
const WIN_R = [3, 4, 5, 6, 7, 8, 9, 10, 11];
const letter = (c) => String.fromCharCode(65 + c);

const on = ref(false);
const shown = ref(MOVES.length);

const toggle = () => {
  on.value = !on.value;
  shown.value = MOVES.length; // entering history mode starts at the latest move
};

const visible = computed(() => MOVES.slice(0, on.value ? shown.value : MOVES.length));
const stones = computed(() =>
  Object.fromEntries(visible.value.map((m) => [`${m.c - WIN_C[0]},${m.r - WIN_R[0]}`, m.v])),
);
const lastKey = computed(() => {
  const m = visible.value.at(-1);
  return m ? `${m.c - WIN_C[0]},${m.r - WIN_R[0]}` : '';
});
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

.stage { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 22px; align-items: start; }
.board-col { max-width: 420px; width: 100%; justify-self: center; }
.buttons { display: flex; justify-content: center; gap: 10px; margin-top: 12px; }
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: inherit; font-size: 13.5px; font-weight: 600;
  padding: 7px 14px; border-radius: 6px; border: 0; cursor: pointer;
  background: #18181b; color: #fff;
}
.btn i { font-size: 12px; }
.btn:disabled { background: #71717a; opacity: 0.6; cursor: default; }

.side { display: flex; flex-direction: column; gap: 12px; }
.panel { border-radius: 8px; border: 1px solid #e4e4e7; background: #fff; overflow: hidden; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); }
.panel-head { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #fafafa; border-bottom: 1px solid #e4e4e7; }
.hist {
  font-family: inherit; font-size: 14px; font-weight: 500;
  padding: 4px 8px; border-radius: 6px; border: 0; cursor: pointer;
  background: transparent; color: #52525b;
  box-shadow: 0 0 0 2px #ef4444;
}
.hist:hover { background: #f4f4f5; color: #18181b; }
.hist.active { background: #27272a; color: #fff; box-shadow: none; }
.exports { font-size: 12px; font-weight: 500; color: #a1a1aa; }
.exports i { font-size: 10px; }
.log { list-style: none; background: #18181b; padding: 8px; height: 230px; overflow-y: auto; }
.log li {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none;
  font-size: 12.5px; color: #a1a1aa; padding: 2px 8px; border-radius: 4px;
}
.log li:last-child { background: #27272a; color: #fff; font-weight: 500; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }

@container (max-width: 600px) {
  .stage { grid-template-columns: 1fr; gap: 16px; }
  .log { height: auto; max-height: 230px; }
}
</style>
