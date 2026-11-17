<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Export: URL and JSON</h1>
      <p class="hint">Press <strong>url</strong> or <strong>json</strong> to see what each one exports for the 8-move game from History Mode.</p>

      <section class="panel">
        <div class="panel-head">
          <span class="hist">History</span>
          <div class="exports">
            <button type="button" :class="['exp', { active: view === 'url' }]" @click="onUrl">
              {{ copied ? 'Copied!' : 'url' }} <i class="pi pi-link" />
            </button>
            <button type="button" :class="['exp', { active: view === 'json' }]" @click="view = 'json'">
              json <i class="pi pi-download" />
            </button>
          </div>
        </div>

        <div class="out">
          <template v-if="view === 'url'">
            <p class="cm">// copied to the clipboard: this page's address plus ?data=</p>
            <p class="url"><span class="q">?data=</span>{{ encoded.slice(0, 96) }}<span class="cm">… ({{ encoded.length.toLocaleString('en-US') }} characters)</span></p>
            <p class="cm gap">// ?data= is base64 of the same JSON as the json button. Opening the link loads that game.</p>
          </template>
          <template v-else-if="view === 'json'">
            <p class="cm">// game.json, downloaded by the json button</p>
            <pre>{{ preview }}</pre>
            <a class="dl" :href="jsonHref" download="game.json"><i class="pi pi-download" /> Download this example</a>
          </template>
          <p v-else class="cm">// pick url or json above</p>
        </div>
      </section>

      <ul class="uses">
        <li><strong>url</strong> shares a game: send the link and the other person opens the same position.</li>
        <li><strong>json</strong> keeps a file. The same format is used for AI test cases and opens in the debug page.</li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

// The game from History Mode, in the store's export shape (exportUrl / exportJson in stores/game.store.ts).
const MOVES = [[9, 8], [8, 8], [8, 7], [10, 9], [9, 7], [8, 6], [7, 7], [10, 7]];
const histories = MOVES.map(([x, y], i) => ({ stone: i % 2 ? 'O' : 'X', coordinate: { x, y } }));
const boardData = Array.from({ length: 19 }, () => Array.from({ length: 19 }, () => ({ stone: '.' })));
for (const h of histories) boardData[h.coordinate.y][h.coordinate.x].stone = h.stone;
const settings = {
  enableCapture: true,
  enableDoubleThreeRestriction: true,
  totalPairCaptured: 5,
  firstMove: 'Player1',
  advantage1: 0,
  advantage2: 0,
  isPlayer2AI: false,
  isDebugTurnLocked: true,
  difficulty: 'hard',
  ai: 'minimax',
};
const data = { boardData, histories, settings, turn: 'X', gameOver: false };

const json = JSON.stringify(data);
const encoded = computed(() => (typeof btoa === 'function' ? btoa(json) : ''));
const jsonHref = `data:text/json;charset=utf-8,${encodeURIComponent(json)}`;

const histLines = histories.map((h) => `    { "stone": "${h.stone}", "coordinate": { "x": ${h.coordinate.x}, "y": ${h.coordinate.y} } }`).join(',\n');
const preview = `{
  "boardData": [ 19 rows of 19 { "stone": "." | "X" | "O" } ],
  "histories": [
${histLines}
  ],
  "settings": { "enableCapture": true, "totalPairCaptured": 5, "firstMove": "Player1", "ai": "minimax", … },
  "turn": "X",
  "gameOver": false
}`;

const view = ref('');
const copied = ref(false);
const onUrl = async () => {
  view.value = 'url';
  try {
    await navigator.clipboard.writeText(`${window.location.origin}/gomoku/game?data=${encoded.value}`);
    copied.value = true;
    setTimeout(() => { copied.value = false; }, 1500);
  } catch {
    copied.value = false;
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
.container { width: 100%; max-width: 720px; margin: 0 auto; }
h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 16px; }

.panel { border-radius: 8px; border: 1px solid #e4e4e7; background: #fff; overflow: hidden; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05); }
.panel-head { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; background: #fafafa; border-bottom: 1px solid #e4e4e7; }
.hist { font-size: 14px; font-weight: 500; color: #52525b; padding: 4px 8px; }
.exports { display: flex; gap: 4px; }
.exp {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: inherit; font-size: 13px; font-weight: 500;
  padding: 5px 10px; border-radius: 6px; border: 0; cursor: pointer;
  background: transparent; color: #52525b; box-shadow: 0 0 0 1.5px #fca5a5;
}
.exp i { font-size: 10px; }
.exp:hover { background: #f4f4f5; color: #18181b; }
.exp.active { background: #27272a; color: #fff; box-shadow: none; }

.out {
  background: #18181b; color: #e4e4e7; padding: 12px 14px; min-height: 150px;
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none; font-size: 12.5px; line-height: 1.6;
}
.out p { overflow-wrap: anywhere; }
.out pre { font: inherit; white-space: pre-wrap; overflow-wrap: anywhere; margin-top: 4px; }
.cm { color: #71717a; }
.q { color: #fbbf24; }
.url { margin-top: 4px; color: #7dd3fc; }
.gap { margin-top: 10px; }
.dl {
  display: inline-flex; align-items: center; gap: 6px; margin-top: 10px;
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 12.5px; font-weight: 600;
  color: #18181b; background: #f4f4f5; padding: 5px 10px; border-radius: 6px; text-decoration: none;
}
.dl i { font-size: 11px; }
.dl:hover { background: #fff; }

.uses { margin: 14px 0 0 18px; display: flex; flex-direction: column; gap: 4px; }
.uses li { font-size: 13.5px; line-height: 1.55; color: #334155; }
</style>
