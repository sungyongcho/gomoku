<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>9-Cell Window and 18-bit Encoding</h1>
      <p class="hint">Pick an axis, or click a point to cycle empty → black → white. The center is the move being scored.</p>

      <div class="picker" role="tablist" aria-label="Axis">
        <button
          v-for="a in AXES"
          :key="a.id"
          type="button"
          role="tab"
          :aria-selected="axis === a.id"
          :class="['pick', { active: axis === a.id }]"
          @click="axis = a.id"
        >{{ a.label }}</button>
      </div>

      <div class="layout">
        <div class="board-col">
        <svg class="board" :viewBox="`0 0 ${BOARD_PX} ${BOARD_PX}`" role="img" aria-label="Board section with the selected window">
          <defs>
            <radialGradient id="d08-wood" cx="50%" cy="50%" r="75%">
              <stop offset="0%" stop-color="#fef9c3" />
              <stop offset="100%" stop-color="#ca8a04" />
            </radialGradient>
            <radialGradient id="d08-black" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#4b5563" />
              <stop offset="50%" stop-color="#111827" />
              <stop offset="100%" stop-color="#000" />
            </radialGradient>
            <radialGradient id="d08-white" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stop-color="#fff" />
              <stop offset="60%" stop-color="#fff" />
              <stop offset="100%" stop-color="#d1d5db" />
            </radialGradient>
            <filter id="d08-shadow" x="-30%" y="-30%" width="160%" height="160%">
              <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#78716c" flood-opacity="0.9" />
            </filter>
          </defs>

          <rect x="0" y="0" :width="BOARD_PX" :height="BOARD_PX" rx="6" fill="url(#d08-wood)" />
          <g stroke="#000" stroke-width="1.5">
            <line v-for="i in N" :key="'h' + i" :x1="pos(0)" :x2="pos(N - 1)" :y1="pos(i - 1)" :y2="pos(i - 1)" />
            <line v-for="i in N" :key="'v' + i" :y1="pos(0)" :y2="pos(N - 1)" :x1="pos(i - 1)" :x2="pos(i - 1)" />
          </g>

          <rect
            :x="pos(C) - capsule.half"
            :y="pos(C) - R"
            :width="capsule.half * 2"
            :height="R * 2"
            :rx="R"
            fill="rgba(239, 68, 68, 0.06)"
            stroke="#ef4444"
            stroke-width="2.5"
            :transform="`rotate(${capsule.angle} ${pos(C)} ${pos(C)})`"
          />

          <g v-for="cell in cells" :key="cell.key">
            <circle
              v-if="cell.v"
              :cx="pos(cell.x)"
              :cy="pos(cell.y)"
              :r="STONE_R"
              :fill="cell.v === 1 ? 'url(#d08-black)' : 'url(#d08-white)'"
              filter="url(#d08-shadow)"
            />
            <rect
              v-if="cell.x !== C || cell.y !== C"
              :x="pos(cell.x) - STEP / 2"
              :y="pos(cell.y) - STEP / 2"
              :width="STEP"
              :height="STEP"
              class="hit"
              @click="cycle(cell.x, cell.y)"
            />
          </g>

          <rect
            :x="pos(C) - STEP * 0.55"
            :y="pos(C) - STEP * 0.55"
            :width="STEP * 1.1"
            :height="STEP * 1.1"
            rx="8"
            fill="none"
            stroke="#2563eb"
            stroke-width="3"
          />
        </svg>
          <div class="board-actions">
            <button type="button" class="act" :disabled="!changed && axis === 'd'" @click="reset">↺ Reset to example</button>
            <button type="button" class="act" @click="clear">Clear stones</button>
          </div>
        </div>

        <div class="side">
          <p class="side-title">Window order</p>
          <div class="strip">
            <div v-for="(w, i) in windowCells" :key="i" :class="['wcell', { center: i === 4 }]">
              <span class="d">{{ i === 4 ? 'd' : i < 4 ? `d−${4 - i}` : `d+${i - 4}` }}</span>
              <span class="slot">
                <span v-if="w === 1" class="stone black" />
                <span v-else-if="w === 2" class="stone white" />
                <span v-else-if="w === 3" class="oob">×</span>
              </span>
              <code :class="['bits', { fixed: i === 4 }]">{{ i === 4 ? '00' : CODE[w] }}</code>
            </div>
          </div>
          <p class="center-note">The center is always written as <code>00</code>: each player has its own lookup table, so the center does not need to say whose stone it is.</p>

          <p class="side-title">2 bits per cell</p>
          <ul class="key">
            <li><span class="slot small" /> <code>00</code> empty</li>
            <li><span class="slot small"><span class="stone black" /></span> <code>01</code> player 1 (black)</li>
            <li><span class="slot small"><span class="stone white" /></span> <code>10</code> player 2 (white)</li>
            <li><span class="slot small"><span class="oob">×</span></span> <code>11</code> off the board</li>
          </ul>

          <p class="side-title">Table index (18 bits, d−4 first)</p>
          <code class="index">{{ indexGrouped }}</code>
          <p class="index-dec">= {{ indexValue.toLocaleString('en-US') }} → <code>patternScoreTablePlayerOne[{{ indexValue }}]</code> for the black center</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const N = 9;
const C = 4;
const STEP = 44;
const MARGIN = 34;
const BOARD_PX = MARGIN * 2 + STEP * (N - 1);
const STONE_R = STEP * 0.4;
const R = STEP * 0.56;
const CODE = ['00', '01', '10', '11'];

const AXES = [
  { id: 'h', label: 'Horizontal', dx: 1, dy: 0 },
  { id: 'v', label: 'Vertical', dx: 0, dy: 1 },
  { id: 'd', label: 'Diagonal', dx: 1, dy: 1 },
  { id: 'a', label: 'Anti-diagonal', dx: 1, dy: -1 },
];

// 0 empty, 1 black (player 1), 2 white (player 2). The center (4,4) is black.
const START = {
  '1,1': 2, '3,3': 1, '6,6': 2, '8,8': 1,
  '2,4': 2, '5,4': 1, '6,4': 1,
  '4,2': 1, '4,7': 2,
  '6,2': 2, '2,6': 1,
};

const makeBoard = (withStones) =>
  Array.from({ length: N }, (_, y) =>
    Array.from({ length: N }, (_, x) => (x === C && y === C ? 1 : withStones ? START[`${x},${y}`] ?? 0 : 0)),
  );
const board = ref(makeBoard(true));
const reset = () => { board.value = makeBoard(true); axis.value = 'd'; };
const clear = () => { board.value = makeBoard(false); };
const changed = computed(() => JSON.stringify(board.value) !== JSON.stringify(makeBoard(true)));
const axis = ref('d');

const pos = (i) => MARGIN + i * STEP;
const cycle = (x, y) => {
  board.value[y][x] = (board.value[y][x] + 1) % 3;
};

const cells = computed(() => board.value.flatMap((row, y) => row.map((v, x) => ({ key: `${x},${y}`, x, y, v }))));
const dir = computed(() => AXES.find((a) => a.id === axis.value));

const capsule = computed(() => {
  const { dx, dy } = dir.value;
  const len = Math.hypot(dx, dy) * STEP * 4;
  return { half: len + R, angle: (Math.atan2(dy, dx) * 180) / Math.PI };
});

const windowCells = computed(() => {
  const { dx, dy } = dir.value;
  return Array.from({ length: 9 }, (_, i) => {
    const x = C + (i - 4) * dx;
    const y = C + (i - 4) * dy;
    if (x < 0 || x >= N || y < 0 || y >= N) return 3;
    return board.value[y][x];
  });
});

const indexBits = computed(() => windowCells.value.map((w, i) => (i === 4 ? '00' : CODE[w])));
const indexGrouped = computed(() => indexBits.value.join(' '));
const indexValue = computed(() => parseInt(indexBits.value.join(''), 2));
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
.container { width: 100%; max-width: 1000px; margin: 0 auto; }

h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 12px; }

.picker { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-bottom: 18px; }
.pick {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.pick:hover { border-color: #94a3b8; }
.pick.active { background: #fee2e2; border-color: #ef4444; color: #991b1b; }

.layout { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 24px; align-items: start; }
.board { width: 100%; height: auto; display: block; max-width: 420px; margin: 0 auto; }
.hit { fill: transparent; cursor: pointer; }
.board-actions { display: flex; justify-content: center; gap: 8px; margin-top: 10px; }
.act {
  font-family: inherit; font-size: 12.5px; font-weight: 600;
  padding: 6px 12px; border-radius: 6px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.act:hover:not(:disabled) { background: #f1f5f9; border-color: #94a3b8; }
.act:disabled { opacity: 0.4; cursor: default; }
.hit:hover { fill: rgba(0, 0, 0, 0.08); }

.side-title { font-size: 13px; font-weight: 700; color: #0f172a; margin: 14px 0 8px; }
.side-title:first-child { margin-top: 0; }

.strip { display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 3px; }
.wcell { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 4px 0; border-radius: 6px; }
.wcell.center { outline: 2.5px solid #2563eb; outline-offset: -1px; }
.d { font-size: 11px; color: #64748b; white-space: nowrap; }
.slot {
  width: 100%; max-width: 34px; aspect-ratio: 1; position: relative;
  background: radial-gradient(circle at center, #fef9c3, #eab308);
  border-radius: 4px; display: flex; align-items: center; justify-content: center;
}
.slot::before { content: ''; position: absolute; left: 0; right: 0; top: 50%; border-top: 1.5px solid #000; }
.slot::after { content: ''; position: absolute; top: 0; bottom: 0; left: 50%; border-left: 1.5px solid #000; }
.slot.small { width: 26px; max-width: 26px; flex: none; }
.stone { position: absolute; inset: 8%; border-radius: 50%; z-index: 1; box-shadow: 0 2px 6px 1px #78716c; }
.stone.black { background: radial-gradient(circle at center, #4b5563, #111827 50%, #000); }
.stone.white { background: radial-gradient(circle at center, #fff, #fff 60%, #d1d5db); }
.oob { position: relative; z-index: 1; font-weight: 800; font-size: 15px; color: #b91c1c; background: #fef2f2; padding: 0 3px; border-radius: 3px; }

code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none; font-size: 12.5px;
}
.bits { font-size: 12px; font-weight: 700; color: #1d4ed8; }
.bits.fixed { color: #64748b; }
.center-note { font-size: 12px; line-height: 1.5; color: #64748b; margin-top: 8px; }

.key { list-style: none; display: grid; grid-template-columns: 1fr 1fr; gap: 8px 12px; }
.key li { display: flex; align-items: center; gap: 8px; font-size: 13px; }

.index {
  display: block; font-size: 14px; font-weight: 700; letter-spacing: 0.02em;
  background: #1e293b; color: #e2e8f0; border-radius: 8px; padding: 10px 12px;
  overflow-wrap: anywhere;
}
.index-dec { font-size: 12.5px; color: #475569; margin-top: 6px; overflow-wrap: anywhere; }

@container (max-width: 720px) {
  .layout { grid-template-columns: 1fr; }
  .key { grid-template-columns: 1fr; }
}
</style>
