<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>D4 Symmetry Augmentation: One Position, Eight Views</h1>
      <p class="hint">Hover a point to see where it lands in every view. Click points on <strong>R0</strong> to edit the position.</p>

      <div class="boards">
        <figure
          v-for="v in VIEWS"
          :key="v.k"
          :class="['view', { picked: picked === v.k, original: v.k === 0 }]"
          @click="picked = v.k"
        >
          <figcaption>
            <span class="name">{{ v.label }}</span>
            <code class="k">k = {{ v.k }}</code>
          </figcaption>
          <svg :viewBox="`0 0 ${PX} ${PX}`" role="img" :aria-label="`${v.label} view`">
            <defs v-if="v.k === 0">
              <radialGradient id="d05-wood" cx="0%" cy="0%" r="140%">
                <stop offset="0%" stop-color="#ca8a04" />
                <stop offset="100%" stop-color="#fde68a" />
              </radialGradient>
              <radialGradient id="d05-black" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#4b5563" />
                <stop offset="50%" stop-color="#111827" />
                <stop offset="100%" stop-color="#000" />
              </radialGradient>
              <radialGradient id="d05-white" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="#fff" />
                <stop offset="60%" stop-color="#fff" />
                <stop offset="100%" stop-color="#d1d5db" />
              </radialGradient>
              <filter id="d05-shadow" x="-30%" y="-30%" width="160%" height="160%">
                <feDropShadow dx="0" dy="1.5" stdDeviation="1.8" flood-color="#78716c" flood-opacity="0.9" />
              </filter>
            </defs>
            <rect x="0" y="0" :width="PX" :height="PX" fill="url(#d05-wood)" />
            <g stroke="#000" stroke-width="1.2">
              <line v-for="i in N" :key="'h' + i" x1="0" :x2="PX" :y1="pos(i - 1)" :y2="pos(i - 1)" />
              <line v-for="i in N" :key="'v' + i" y1="0" :y2="PX" :x1="pos(i - 1)" :x2="pos(i - 1)" />
            </g>
            <circle
              v-if="hoverPoint"
              :cx="pos(map(hoverPoint, v.k).x)"
              :cy="pos(map(hoverPoint, v.k).y)"
              :r="STEP * 0.48"
              class="trace"
            />
            <circle
              v-for="s in stonesIn(v.k)"
              :key="s.key"
              :cx="pos(s.x)"
              :cy="pos(s.y)"
              :r="STEP * 0.38"
              :fill="s.v === 1 ? 'url(#d05-black)' : 'url(#d05-white)'"
              filter="url(#d05-shadow)"
            />
            <rect
              v-for="p in POINTS"
              :key="'hit' + p.key"
              :x="pos(p.x) - STEP / 2"
              :y="pos(p.y) - STEP / 2"
              :width="STEP"
              :height="STEP"
              :class="['hit', { editable: v.k === 0 }]"
              @mouseenter="hoverPoint = inverse(p, v.k)"
              @mouseleave="hoverPoint = null"
              @click.stop="v.k === 0 ? cycle(p) : (picked = v.k)"
            />
          </svg>
        </figure>
      </div>

      <div class="panel">
        <p class="panel-title">{{ VIEWS[picked].label }}: {{ VIEWS[picked].how }}</p>
        <p class="panel-body">
          <code>_apply_symmetry</code> draws <code>k = np.random.randint(0, 8)</code> each time a sample is read.
          <template v-if="picked >= 4"><code>k ≥ 4</code> first flips the width axis, then </template>
          <template v-else>It </template>
          rotates by <code>np.rot90(k={{ picked % 4 }})</code>. The policy target π gets the same transform, so the best move moves with the stones.
        </p>
        <div class="panel-actions">
          <button type="button" class="act" :disabled="!changed" @click="reset">↺ Reset position</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const N = 5;
const STEP = 40;
const PX = STEP * (N + 1);
const pos = (i) => STEP * (i + 1);

const VIEWS = [
  { k: 0, label: 'R0', how: 'the stored position' },
  { k: 1, label: 'R90', how: 'rotated 90° counter-clockwise' },
  { k: 2, label: 'R180', how: 'rotated 180°' },
  { k: 3, label: 'R270', how: 'rotated 270° counter-clockwise' },
  { k: 4, label: 'Flip+R0', how: 'mirrored left to right' },
  { k: 5, label: 'Flip+R90', how: 'mirrored, then rotated 90°' },
  { k: 6, label: 'Flip+R180', how: 'mirrored, then rotated 180°' },
  { k: 7, label: 'Flip+R270', how: 'mirrored, then rotated 270°' },
];

const POINTS = Array.from({ length: N * N }, (_, i) => ({ x: i % N, y: Math.floor(i / N), key: i }));

// 1 black, 2 white; coordinates are (column, row) on the R0 board.
const START = { '0,0': 1, '2,1': 2, '3,1': 1, '1,2': 1, '4,2': 2, '0,3': 2, '3,4': 2, '4,4': 1 };
const stones = ref({ ...START });
const picked = ref(0);
const hoverPoint = ref(null);

// Same order as the code: flip the width axis when k >= 4, then np.rot90 (counter-clockwise) k % 4 times.
const map = ({ x, y }, k) => {
  let c = k >= 4 ? N - 1 - x : x;
  let r = y;
  for (let i = 0; i < k % 4; i++) {
    const nr = N - 1 - c;
    c = r;
    r = nr;
  }
  return { x: c, y: r };
};
const inverse = (p, k) => POINTS.find((q) => {
  const m = map(q, k);
  return m.x === p.x && m.y === p.y;
});

const stonesIn = (k) =>
  Object.entries(stones.value).map(([key, v]) => {
    const [x, y] = key.split(',').map(Number);
    const m = map({ x, y }, k);
    return { key, v, x: m.x, y: m.y };
  });

const cycle = ({ x, y }) => {
  const key = `${x},${y}`;
  const next = ((stones.value[key] ?? 0) + 1) % 3;
  const copy = { ...stones.value };
  if (next) copy[key] = next;
  else delete copy[key];
  stones.value = copy;
};
const changed = computed(() => JSON.stringify(stones.value) !== JSON.stringify(START));
const reset = () => { stones.value = { ...START }; };
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
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 16px; }

.boards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.view { cursor: pointer; border-radius: 10px; padding: 6px; border: 2px solid transparent; transition: border-color 0.15s, background 0.15s; }
.view:hover { background: #f1f5f9; }
.view.picked { border-color: #2563eb; background: #eff6ff; }
figcaption { display: flex; justify-content: space-between; align-items: baseline; margin: 0 2px 6px; }
.name { font-size: 15px; font-weight: 700; color: #0f172a; }
.original .name::after { content: ' · edit me'; font-size: 11px; font-weight: 600; color: #64748b; }
.k { font-size: 11.5px; color: #64748b; }
svg { width: 100%; height: auto; display: block; border-radius: 3px; box-shadow: 0 0 0 1.5px #1c1917; }

.hit { fill: transparent; }
.hit.editable { cursor: pointer; }
.hit.editable:hover { fill: rgba(0, 0, 0, 0.1); }
.trace { fill: rgba(239, 68, 68, 0.18); stroke: #ef4444; stroke-width: 2.5; }

.panel {
  margin-top: 16px; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px;
}
.panel-title { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 4px; }
.panel-body { font-size: 13px; line-height: 1.6; color: #475569; }
.panel-actions { margin-top: 10px; }
.act {
  font-family: inherit; font-size: 12.5px; font-weight: 600;
  padding: 6px 12px; border-radius: 6px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.act:hover:not(:disabled) { background: #f1f5f9; border-color: #94a3b8; }
.act:disabled { opacity: 0.4; cursor: default; }

code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none; font-size: 0.9em;
}

@container (max-width: 620px) {
  .boards { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
}
</style>
