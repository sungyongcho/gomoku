<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Undo a Turn / Undo a Move</h1>
      <p class="hint">Pick a mode and press the undo button. Dashed rings mark the stones the last undo took back.</p>

      <div class="picker" role="tablist" aria-label="Game mode">
        <button
          v-for="m in MODES"
          :key="m.id"
          type="button"
          role="tab"
          :aria-selected="mode === m.id"
          :class="['pick', { active: mode === m.id }]"
          @click="choose(m.id)"
        >
          {{ m.name }}
        </button>
      </div>

      <div class="stage">
        <div class="board-col">
          <RuleBoard
            :cols="7"
            :rows="5"
            :stones="stones"
            :last="lastKey"
            :removed="undone"
            :label="`${current.name}, ${count} moves on the board`"
          />
          <div class="buttons">
            <button type="button" class="btn" :disabled="!canUndo" @click="undo"><i class="pi pi-undo" /> {{ current.button }}</button>
            <button type="button" class="btn" :disabled="count === MOVES.length" @click="reset"><i class="pi pi-replay" /> Reset example</button>
          </div>
        </div>

        <div class="side">
          <ul class="log">
            <li v-for="(m, i) in MOVES.slice(0, count)" :key="i">
              [{{ m.v === 1 ? 'Black' : 'White' }}]: {{ who(m) }}
            </li>
            <li v-if="!count" class="empty">No moves yet</li>
          </ul>
          <p class="text" v-html="current.text" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

const MOVES = [
  { c: 3, r: 2, v: 1 },
  { c: 3, r: 1, v: 2 },
  { c: 4, r: 3, v: 1 },
  { c: 2, r: 3, v: 2 },
  { c: 2, r: 1, v: 1 },
  { c: 4, r: 1, v: 2 },
];

const MODES = [
  {
    id: 'ai',
    name: 'Player vs AI',
    button: 'Undo a turn',
    text: '<strong>Undo a turn</strong> takes back your last move and the AI\'s reply together, so it is your turn again. It works only when the last move was the AI\'s (White).',
  },
  {
    id: 'pvp',
    name: 'Player vs Player',
    button: 'Undo a move',
    text: '<strong>Undo a move</strong> takes back only the most recent move, whoever played it.',
  },
];

const mode = ref('ai');
const count = ref(MOVES.length);
const undone = ref([]);
const current = computed(() => MODES.find((m) => m.id === mode.value));

const who = (m) => (mode.value === 'ai' ? (m.v === 1 ? 'you' : 'AI') : (m.v === 1 ? 'Player 1' : 'Player 2'));

const stones = computed(() => Object.fromEntries(MOVES.slice(0, count.value).map((m) => [`${m.c},${m.r}`, m.v])));
const lastKey = computed(() => {
  const m = MOVES[count.value - 1];
  return m ? `${m.c},${m.r}` : '';
});

// Mirrors canUndoTurn / deleteLastTurn in stores/game.store.ts.
const canUndo = computed(() => {
  if (!count.value) return false;
  return mode.value === 'pvp' || MOVES[count.value - 1].v === 2;
});
const undo = () => {
  const n = mode.value === 'ai' && count.value >= 2 ? 2 : 1;
  undone.value = MOVES.slice(count.value - n, count.value).map(({ c, r }) => ({ c, r }));
  count.value -= n;
};
const reset = () => {
  count.value = MOVES.length;
  undone.value = [];
};
const choose = (id) => {
  mode.value = id;
  reset();
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
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 12px; }

.picker { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.pick {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 8px 18px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155; transition: all 0.15s;
}
.pick:hover { border-color: #94a3b8; }
.pick.active { background: #1e293b; border-color: #1e293b; color: #fff; }

.stage { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 22px; align-items: start; }
.board-col { max-width: 400px; width: 100%; justify-self: center; }
.buttons { display: flex; justify-content: center; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
.btn {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: inherit; font-size: 13.5px; font-weight: 600;
  padding: 7px 14px; border-radius: 6px; border: 0; cursor: pointer;
  background: #18181b; color: #fff;
}
.btn i { font-size: 12px; }
.btn:disabled { background: #71717a; opacity: 0.6; cursor: default; }

.side { display: flex; flex-direction: column; gap: 12px; }
.log { list-style: none; background: #18181b; border-radius: 8px; padding: 8px; min-height: 120px; }
.log li {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none;
  font-size: 12.5px; color: #a1a1aa; padding: 2px 8px; border-radius: 4px;
}
.log li:last-child { background: #27272a; color: #fff; font-weight: 500; }
.log li.empty { background: none; color: #71717a; font-weight: 400; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }

@container (max-width: 600px) {
  .stage { grid-template-columns: 1fr; gap: 16px; }
}
</style>
