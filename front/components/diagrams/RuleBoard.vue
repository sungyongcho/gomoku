<template>
  <svg :viewBox="`0 0 ${GX + W} ${GY + H}`" class="rule-board" role="img" :aria-label="label">
    <defs>
      <radialGradient :id="`${uid}-wood`" cx="50%" cy="50%" r="75%">
        <stop offset="0%" stop-color="#fef9c3" />
        <stop offset="100%" stop-color="#ca8a04" />
      </radialGradient>
      <!-- History mode greys the board out, as GoBoard does (stone-100 to stone-600). -->
      <radialGradient :id="`${uid}-history`" cx="50%" cy="50%" r="75%">
        <stop offset="0%" stop-color="#f5f5f4" />
        <stop offset="100%" stop-color="#57534e" />
      </radialGradient>
      <radialGradient :id="`${uid}-black`" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#4b5563" />
        <stop offset="50%" stop-color="#111827" />
        <stop offset="100%" stop-color="#000" />
      </radialGradient>
      <radialGradient :id="`${uid}-white`" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#fff" />
        <stop offset="60%" stop-color="#fff" />
        <stop offset="100%" stop-color="#d1d5db" />
      </radialGradient>
      <filter :id="`${uid}-shadow`" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#78716c" flood-opacity="0.9" />
      </filter>
    </defs>

    <!-- A cut-out of a larger board: grid lines run off every edge. -->
    <rect x="0" y="0" :width="GX + W" :height="GY + H" :fill="`url(#${uid}-${tone})`" class="wood" />
    <g v-if="colLabels.length" class="coord">
      <text v-for="(t, i) in colLabels" :key="'cl' + i" :x="GX + pos(i)" :y="GY * 0.62">{{ t }}</text>
      <text v-for="(t, i) in rowLabels" :key="'rl' + i" :x="GX * 0.5" :y="GY + pos(i) + 5">{{ t }}</text>
    </g>
    <g :transform="`translate(${GX} ${GY})`">
    <g stroke="#000" stroke-width="1.5">
      <line v-for="c in cols" :key="'v' + c" :x1="pos(c - 1)" :x2="pos(c - 1)" y1="0" :y2="H" />
      <line v-for="r in rows" :key="'h' + r" :y1="pos(r - 1)" :y2="pos(r - 1)" x1="0" :x2="W" />
    </g>

    <rect
      v-for="(seg, i) in segments"
      :key="'seg' + i"
      :x="seg.cx - seg.half"
      :y="seg.cy - CAP_R"
      :width="seg.half * 2"
      :height="CAP_R * 2"
      :rx="CAP_R"
      :fill="seg.fill"
      :stroke="seg.color"
      stroke-width="2.5"
      :transform="`rotate(${seg.angle} ${seg.cx} ${seg.cy})`"
      :class="{ dim: seg.dim }"
    />

    <circle
      v-for="s in removed"
      :key="'rm' + s.c + ',' + s.r"
      :cx="pos(s.c)"
      :cy="pos(s.r)"
      :r="STONE_R"
      class="removed"
    />

    <circle
      v-for="s in stoneList"
      :key="s.key"
      :cx="pos(s.c)"
      :cy="pos(s.r)"
      :r="STONE_R"
      :fill="s.v === 1 ? `url(#${uid}-black)` : `url(#${uid}-white)`"
      :filter="`url(#${uid}-shadow)`"
      :class="{ fresh: s.key === fresh }"
    />
    <circle v-if="lastStone" :cx="pos(lastStone.c)" :cy="pos(lastStone.r)" :r="STONE_R - 1" class="last" />
    <rect
      v-for="p in hitPoints"
      :key="'hit' + p.key"
      :x="pos(p.c) - STEP / 2"
      :y="pos(p.r) - STEP / 2"
      :width="STEP"
      :height="STEP"
      class="hit"
      @click="emit('point', p)"
      @contextmenu.prevent="emit('point', p)"
    />

    <g v-if="ghost" :class="['ghost', { clickable: ghostClickable }]" @click="ghostClickable && emit('play')">
      <circle :cx="pos(ghost.c)" :cy="pos(ghost.r)" :r="STONE_R" :class="ghost.v === 1 ? 'ghost-black' : 'ghost-white'" />
      <g v-if="ghost.forbidden" class="cross">
        <line :x1="pos(ghost.c) - X_R" :y1="pos(ghost.r) - X_R" :x2="pos(ghost.c) + X_R" :y2="pos(ghost.r) + X_R" />
        <line :x1="pos(ghost.c) + X_R" :y1="pos(ghost.r) - X_R" :x2="pos(ghost.c) - X_R" :y2="pos(ghost.r) + X_R" />
      </g>
      <circle v-if="ghostClickable" :cx="pos(ghost.c)" :cy="pos(ghost.r)" :r="STONE_R + 4" class="pulse" />
    </g>
    </g>
  </svg>
</template>

<script setup>
import { computed, useId } from 'vue';

const props = defineProps({
  cols: { type: Number, required: true },
  rows: { type: Number, required: true },
  // { 'c,r': 1 | 2 }, 1 black, 2 white
  stones: { type: Object, default: () => ({}) },
  // { c, r, v, forbidden? } a stone that is about to be played, drawn faded
  ghost: { type: Object, default: null },
  ghostClickable: { type: Boolean, default: false },
  // [{ c, r }] stones that were just captured, drawn as dashed rings
  removed: { type: Array, default: () => [] },
  // [{ from: [c, r], to: [c, r], color, dim? }] highlighted lines of stones
  lines: { type: Array, default: () => [] },
  fresh: { type: String, default: '' },
  label: { type: String, default: 'Board position' },
  // 'wood' for play, 'history' for the grey history-mode board
  tone: { type: String, default: 'wood' },
  // 'c,r' of the last move, drawn with the game's red ring
  last: { type: String, default: '' },
  // optional coordinate labels along the top and the left edge
  colLabels: { type: Array, default: () => [] },
  rowLabels: { type: Array, default: () => [] },
  // [{ c, r }] points that emit 'point' when clicked or right-clicked
  hits: { type: Array, default: () => [] },
});
const emit = defineEmits(['play', 'point']);

const uid = `rb${useId().replace(/[^a-zA-Z0-9]/g, '')}`;

const STEP = 48;
const STONE_R = STEP * 0.4;
const CAP_R = STEP * 0.5;
const X_R = STEP * 0.2;
const W = computed(() => STEP * props.cols);
const H = computed(() => STEP * props.rows);
const pos = (i) => STEP * (i + 0.5);
const GX = computed(() => (props.rowLabels.length ? STEP * 0.7 : 0));
const GY = computed(() => (props.colLabels.length ? STEP * 0.6 : 0));
const lastStone = computed(() => {
  if (!props.last || !props.stones[props.last]) return null;
  const [c, r] = props.last.split(',').map(Number);
  return { c, r };
});
const hitPoints = computed(() => props.hits.map((p) => ({ ...p, key: `${p.c},${p.r}` })));

const stoneList = computed(() =>
  Object.entries(props.stones).map(([key, v]) => {
    const [c, r] = key.split(',').map(Number);
    return { key, c, r, v };
  }),
);

const segments = computed(() =>
  props.lines.map(({ from, to, color, dim }) => {
    const x1 = pos(from[0]);
    const y1 = pos(from[1]);
    const x2 = pos(to[0]);
    const y2 = pos(to[1]);
    return {
      cx: (x1 + x2) / 2,
      cy: (y1 + y2) / 2,
      half: Math.hypot(x2 - x1, y2 - y1) / 2 + CAP_R,
      angle: (Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI,
      color,
      fill: `${color}14`,
      dim,
    };
  }),
);
</script>

<style scoped>
.rule-board { width: 100%; height: auto; display: block; border-radius: 6px; box-shadow: 0 0 0 1.5px #1c1917; }
.removed { fill: rgba(220, 38, 38, 0.08); stroke: #dc2626; stroke-width: 2; stroke-dasharray: 5 4; }
.ghost-black { fill: #000; opacity: 0.4; }
.ghost-white { fill: #fff; opacity: 0.75; stroke: #a8a29e; stroke-width: 1.5; }
.cross line { stroke: #dc2626; stroke-width: 4; stroke-linecap: round; }
.ghost.clickable { cursor: pointer; }
.ghost:not(.clickable), .removed, .last { pointer-events: none; }
.ghost.clickable:hover .ghost-black { opacity: 0.6; }
.pulse { fill: none; stroke: #2563eb; stroke-width: 2.5; animation: pulse 1.6s ease-in-out infinite; }
.dim { opacity: 0.25; }
.last { fill: none; stroke: #ef4444; stroke-width: 4; }
.hit { fill: transparent; cursor: pointer; }
.hit:hover { fill: rgba(0, 0, 0, 0.08); }
.coord text { font: 700 15px 'Helvetica Neue', Helvetica, Arial, sans-serif; fill: #1c1917; text-anchor: middle; }
.wood { transition: fill 0.3s; }
rect { transition: opacity 0.15s; }
.fresh { animation: drop 0.25s ease-out; transform-box: fill-box; transform-origin: center; }
@keyframes pulse { 0%, 100% { opacity: 0.25; } 50% { opacity: 1; } }
@keyframes drop { from { transform: scale(1.25); opacity: 0; } to { transform: scale(1); opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
  .pulse, .fresh { animation: none; }
}
</style>
