<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Board-Translation Augmentation (<code>_apply_translation</code>)</h1>

      <div class="controls-panel">
        <div class="btn-row" role="group" aria-label="Board presets">
          <button
            v-for="p in PRESETS"
            :key="p.id"
            type="button"
            class="cfg-btn"
            :class="{ active: presetId === p.id }"
            @click="selectPreset(p.id)"
          >
            {{ p.label }}
          </button>
        </div>

        <div class="shift-controls">
          <button type="button" class="action-btn" @click="applyRandomD4">
            D4 Symmetry (center fixed)
          </button>
          <button
            type="button"
            class="action-btn primary"
            :disabled="isSkipped"
            @click="randomShift"
          >
            Random Valid Shift
          </button>
          <div class="nudge-group">
            <span><code>Δx</code>:</span>
            <button type="button" class="step-btn" :disabled="isSkipped || dx <= dxRange[0]" @click="dx--">−</button>
            <strong>{{ fmtShift(dx) }}</strong>
            <button type="button" class="step-btn" :disabled="isSkipped || dx >= dxRange[1]" @click="dx++">+</button>
          </div>
          <div class="nudge-group">
            <span><code>Δy</code>:</span>
            <button type="button" class="step-btn" :disabled="isSkipped || dy <= dyRange[0]" @click="dy--">−</button>
            <strong>{{ fmtShift(dy) }}</strong>
            <button type="button" class="step-btn" :disabled="isSkipped || dy >= dyRange[1]" @click="dy++">+</button>
          </div>
          <button type="button" class="step-btn reset-btn" @click="resetView">Reset</button>
        </div>
      </div>

      <div class="status-bar" :class="statusClass">
        <template v-if="isSkipped">
          <strong>Skipped (returned unchanged):</strong> A stone lies on the outer line (row <code>{{ minRow }}..{{ maxRow }}</code>, col <code>{{ minCol }}..{{ maxCol }}</code>), so <code>_apply_translation</code> leaves the sample in place.
        </template>
        <template v-else>
          <strong>Valid shift range:</strong>
          <span> <code>Δy ∈ [{{ dxRangeStr(dyRange) }}]</code>, <code>Δx ∈ [{{ dxRangeStr(dxRange) }}]</code> (keeps stones off lines 0 and {{ GRID - 1 }}).</span>
          <template v-if="droppedMass > 0">
            <strong>Wrapped policy dropped:</strong> {{ (droppedMass * 100).toFixed(0) }}% entered the red <code>crossed</code> edge strip, leaving <code>kept = {{ (keptMass * 100).toFixed(0) }}%</code>; remaining targets divide by <code>kept</code>.
          </template>
          <template v-else>
            All policy mass stays on-board (<code>kept = 100%</code>); D4 index: <code>k={{ d4Index }}</code>.
          </template>
        </template>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Comparison of board before and after translation augmentation on an illustrative 9x9 grid"
      >
        <!-- Toy disclaimer -->
        <rect x="16" y="8" width="748" height="26" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="25" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative 9×9 toy grid showing the 19×19 rules in dataset.py (outer ring shaded) — not a training sample
        </text>

        <!-- Left panel: After D4 (before translation) -->
        <text x="185" y="54" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
          1. After D4 Symmetry (k = {{ d4Index }})
        </text>
        <text x="185" y="70" text-anchor="middle" font-size="11" fill="#64748b">
          Center &amp; edge distances preserved
        </text>

        <g :transform="`translate(${LEFT_X}, ${BOARD_Y})`">
          <!-- Outer ring shading -->
          <rect x="0" y="0" :width="BOARD_PX" :height="BOARD_PX" fill="#fef2f2" opacity="0.55" />
          <rect :x="CELL" :y="CELL" :width="BOARD_PX - 2 * CELL" :height="BOARD_PX - 2 * CELL" fill="#ffffff" />

          <!-- Grid lines -->
          <g v-for="i in GRID" :key="'lg-' + i">
            <line :x1="cellCenter(i - 1)" :y1="cellCenter(0)" :x2="cellCenter(i - 1)" :y2="cellCenter(GRID - 1)" stroke="#cbd5e1" stroke-width="1" />
            <line :x1="cellCenter(0)" :y1="cellCenter(i - 1)" :x2="cellCenter(GRID - 1)" :y2="cellCenter(i - 1)" stroke="#cbd5e1" stroke-width="1" />
          </g>

          <!-- Center dot -->
          <circle :cx="cellCenter(4)" :cy="cellCenter(4)" r="3" fill="#94a3b8" />

          <!-- Stone bounding box -->
          <rect
            :x="minCol * CELL + 2"
            :y="minRow * CELL + 2"
            :width="(maxCol - minCol + 1) * CELL - 4"
            :height="(maxRow - minRow + 1) * CELL - 4"
            fill="none"
            :stroke="isSkipped ? '#dc2626' : '#2563eb'"
            stroke-width="1.5"
            stroke-dasharray="4 3"
          />

          <!-- Policy targets before translation -->
          <g v-for="(p, idx) in transformed.policy" :key="'lp-' + idx">
            <rect
              :x="p.x * CELL + 3"
              :y="p.y * CELL + 3"
              :width="CELL - 6"
              :height="CELL - 6"
              rx="4"
              fill="#dbeafe"
              stroke="#2563eb"
              stroke-width="1.2"
            />
            <text
              :x="cellCenter(p.x)"
              :y="cellCenter(p.y) + 4"
              text-anchor="middle"
              font-size="10.5"
              font-weight="700"
              fill="#1e40af"
            >
              {{ Math.round(p.w * 100) }}%
            </text>
          </g>

          <!-- Stones -->
          <circle
            v-for="(s, idx) in transformed.stones"
            :key="'ls-' + idx"
            :cx="cellCenter(s.x)"
            :cy="cellCenter(s.y)"
            r="10.5"
            :fill="s.color === 'B' ? '#0f172a' : '#ffffff'"
            stroke="#0f172a"
            stroke-width="1.8"
          />
        </g>

        <!-- Center arrow -->
        <line x1="348" y1="210" x2="432" y2="210" stroke="#475569" stroke-width="2" />
        <polygon points="432,205 442,210 432,215" fill="#475569" />
        <text x="390" y="196" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          {{ isSkipped ? 'No shift' : `roll(${fmtShift(dy)}, ${fmtShift(dx)})` }}
        </text>
        <text x="390" y="230" text-anchor="middle" font-size="10.5" fill="#64748b">
          {{ isSkipped ? 'outer stone' : `kept ${(keptMass * 100).toFixed(0)}%` }}
        </text>

        <!-- Right panel: After _apply_translation -->
        <text x="595" y="54" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
          2. After _apply_translation
        </text>
        <text x="595" y="70" text-anchor="middle" font-size="11" fill="#64748b">
          {{ isSkipped ? 'Unchanged (outer line guard)' : 'Wrapped edges masked & target renormalized' }}
        </text>

        <g :transform="`translate(${RIGHT_X}, ${BOARD_Y})`">
          <rect x="0" y="0" :width="BOARD_PX" :height="BOARD_PX" fill="#ffffff" />

          <!-- Highlight crossed strips when shifted -->
          <rect
            v-if="!isSkipped && dy > 0"
            x="0"
            y="0"
            :width="BOARD_PX"
            :height="dy * CELL"
            fill="#fee2e2"
            opacity="0.85"
          />
          <rect
            v-if="!isSkipped && dy < 0"
            x="0"
            :y="(GRID + dy) * CELL"
            :width="BOARD_PX"
            :height="-dy * CELL"
            fill="#fee2e2"
            opacity="0.85"
          />
          <rect
            v-if="!isSkipped && dx > 0"
            x="0"
            y="0"
            :width="dx * CELL"
            :height="BOARD_PX"
            fill="#fee2e2"
            opacity="0.85"
          />
          <rect
            v-if="!isSkipped && dx < 0"
            :x="(GRID + dx) * CELL"
            y="0"
            :width="-dx * CELL"
            :height="BOARD_PX"
            fill="#fee2e2"
            opacity="0.85"
          />

          <!-- Grid lines -->
          <g v-for="i in GRID" :key="'rg-' + i">
            <line :x1="cellCenter(i - 1)" :y1="cellCenter(0)" :x2="cellCenter(i - 1)" :y2="cellCenter(GRID - 1)" stroke="#cbd5e1" stroke-width="1" />
            <line :x1="cellCenter(0)" :y1="cellCenter(i - 1)" :x2="cellCenter(GRID - 1)" :y2="cellCenter(i - 1)" stroke="#cbd5e1" stroke-width="1" />
          </g>

          <circle :cx="cellCenter(4)" :cy="cellCenter(4)" r="3" fill="#94a3b8" />

          <!-- Wrapped/dropped policy targets -->
          <g v-for="(p, idx) in droppedPolicies" :key="'rp-drop-' + idx">
            <rect
              :x="p.rx * CELL + 3"
              :y="p.ry * CELL + 3"
              :width="CELL - 6"
              :height="CELL - 6"
              rx="4"
              fill="#fecaca"
              stroke="#dc2626"
              stroke-width="1.4"
              stroke-dasharray="3 2"
            />
            <text
              :x="cellCenter(p.rx)"
              :y="cellCenter(p.ry) + 4"
              text-anchor="middle"
              font-size="10.5"
              font-weight="700"
              fill="#991b1b"
            >
              0%
            </text>
          </g>

          <!-- Kept & renormalized policy targets -->
          <g v-for="(p, idx) in keptPolicies" :key="'rp-keep-' + idx">
            <rect
              :x="p.nx * CELL + 3"
              :y="p.ny * CELL + 3"
              :width="CELL - 6"
              :height="CELL - 6"
              rx="4"
              fill="#dcfce7"
              stroke="#16a34a"
              stroke-width="1.3"
            />
            <text
              :x="cellCenter(p.nx)"
              :y="cellCenter(p.ny) + 4"
              text-anchor="middle"
              :font-size="Math.round(p.renorm * 100) >= 100 ? 9.5 : 10.5"
              font-weight="700"
              fill="#166534"
            >
              {{ Math.round(p.renorm * 100) }}%
            </text>
          </g>

          <!-- Shifted stones -->
          <circle
            v-for="(s, idx) in shiftedStones"
            :key="'rs-' + idx"
            :cx="cellCenter(s.x)"
            :cy="cellCenter(s.y)"
            r="10.5"
            :fill="s.color === 'B' ? '#0f172a' : '#ffffff'"
            stroke="#0f172a"
            stroke-width="1.8"
          />
        </g>

        <!-- Bottom Legend -->
        <rect x="16" y="356" width="748" height="26" rx="5" fill="#ffffff" stroke="#e2e8f0" stroke-width="1" />
        <rect x="28" y="364" width="12" height="10" rx="2" fill="#dbeafe" stroke="#2563eb" />
        <text x="45" y="373" font-size="11" fill="#334155">Input policy target</text>
        <rect x="162" y="364" width="12" height="10" rx="2" fill="#dcfce7" stroke="#16a34a" />
        <text x="179" y="373" font-size="11" fill="#166534">Renormalized target (π / kept)</text>
        <rect x="355" y="364" width="12" height="10" rx="2" fill="#fee2e2" stroke="#dc2626" />
        <text x="372" y="373" font-size="11" fill="#991b1b">
          crossed edge strip (wrapped targets &amp; forbidden plane zeroed)
        </text>
      </svg>

      <p class="source-note">
        Reflects <code>_apply_symmetry</code> and <code>_apply_translation</code> in <code>learning/dataset.py</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const GRID = 9;
const CELL = 29;
const BOARD_PX = GRID * CELL; // 261
const LEFT_X = 54;
const RIGHT_X = 465;
const BOARD_Y = 82;

const PRESETS = [
  {
    id: 'inner',
    label: 'Inner cluster (shiftable)',
    defaultShift: { dx: 3, dy: 2 },
    stones: [
      { x: 1, y: 1, color: 'B' },
      { x: 2, y: 1, color: 'B' },
      { x: 3, y: 1, color: 'B' },
      { x: 2, y: 2, color: 'W' },
      { x: 3, y: 2, color: 'W' },
    ],
    policy: [
      { x: 4, y: 1, w: 0.7 },
      { x: 1, y: 2, w: 0.3 },
    ],
  },
  {
    id: 'edge-wrap',
    label: 'Policy wraps off-board',
    defaultShift: { dx: 2, dy: 2 },
    stones: [
      { x: 2, y: 2, color: 'B' },
      { x: 3, y: 2, color: 'B' },
      { x: 4, y: 2, color: 'B' },
      { x: 3, y: 3, color: 'W' },
      { x: 4, y: 3, color: 'W' },
    ],
    policy: [
      { x: 5, y: 2, w: 0.6 },
      { x: 7, y: 7, w: 0.4 },
    ],
  },
  {
    id: 'outer-skip',
    label: 'Outer-line stone (skipped)',
    defaultShift: { dx: 0, dy: 0 },
    stones: [
      { x: 0, y: 2, color: 'B' },
      { x: 1, y: 2, color: 'B' },
      { x: 2, y: 2, color: 'W' },
      { x: 2, y: 3, color: 'W' },
    ],
    policy: [
      { x: 3, y: 2, w: 0.8 },
      { x: 1, y: 3, w: 0.2 },
    ],
  },
];

const presetId = ref('inner');
const d4Index = ref(0);
const dx = ref(3);
const dy = ref(2);

const activePreset = computed(
  () => PRESETS.find((p) => p.id === presetId.value) ?? PRESETS[0]
);

function transformCoord(x, y, k) {
  let nx = x;
  let ny = y;
  let r = k;
  if (r >= 4) {
    nx = GRID - 1 - nx;
    r -= 4;
  }
  for (let i = 0; i < r; i++) {
    // 90 deg CCW like np.rot90(axes=(0,1))
    const tx = ny;
    const ty = GRID - 1 - nx;
    nx = tx;
    ny = ty;
  }
  return { x: nx, y: ny };
}

const transformed = computed(() => ({
  stones: activePreset.value.stones.map((s) => ({
    ...transformCoord(s.x, s.y, d4Index.value),
    color: s.color,
  })),
  policy: activePreset.value.policy.map((p) => ({
    ...transformCoord(p.x, p.y, d4Index.value),
    w: p.w,
  })),
}));

const minRow = computed(() => Math.min(...transformed.value.stones.map((s) => s.y)));
const maxRow = computed(() => Math.max(...transformed.value.stones.map((s) => s.y)));
const minCol = computed(() => Math.min(...transformed.value.stones.map((s) => s.x)));
const maxCol = computed(() => Math.max(...transformed.value.stones.map((s) => s.x)));

const isSkipped = computed(
  () =>
    minRow.value === 0 ||
    minCol.value === 0 ||
    maxRow.value === GRID - 1 ||
    maxCol.value === GRID - 1
);

// Matches np.random.randint(1 - rows[0], h - 1 - rows[-1]) where upper bound is exclusive
const dyRange = computed(() => [1 - minRow.value, GRID - 2 - maxRow.value]);
const dxRange = computed(() => [1 - minCol.value, GRID - 2 - maxCol.value]);

function clampShift() {
  if (isSkipped.value) {
    dx.value = 0;
    dy.value = 0;
    return;
  }
  dx.value = Math.min(dxRange.value[1], Math.max(dxRange.value[0], dx.value));
  dy.value = Math.min(dyRange.value[1], Math.max(dyRange.value[0], dy.value));
}

const shiftedStones = computed(() => {
  if (isSkipped.value) return transformed.value.stones;
  return transformed.value.stones.map((s) => ({
    x: s.x + dx.value,
    y: s.y + dy.value,
    color: s.color,
  }));
});

function wrapIdx(v) {
  return ((v % GRID) + GRID) % GRID;
}

const evaluatedPolicies = computed(() => {
  const shiftX = isSkipped.value ? 0 : dx.value;
  const shiftY = isSkipped.value ? 0 : dy.value;
  return transformed.value.policy.map((p) => {
    const rawX = p.x + shiftX;
    const rawY = p.y + shiftY;
    const nx = wrapIdx(rawX);
    const ny = wrapIdx(rawY);
    const crossed =
      (shiftY > 0 && ny < shiftY) ||
      (shiftY < 0 && ny >= GRID + shiftY) ||
      (shiftX > 0 && nx < shiftX) ||
      (shiftX < 0 && nx >= GRID + shiftX);
    return { ...p, nx, ny, rx: nx, ry: ny, crossed };
  });
});

const keptMass = computed(() => {
  const sum = evaluatedPolicies.value
    .filter((p) => !p.crossed)
    .reduce((acc, p) => acc + p.w, 0);
  return sum > 0 ? sum : 1;
});

const droppedMass = computed(() =>
  evaluatedPolicies.value
    .filter((p) => p.crossed)
    .reduce((acc, p) => acc + p.w, 0)
);

const keptPolicies = computed(() =>
  evaluatedPolicies.value
    .filter((p) => !p.crossed)
    .map((p) => ({ ...p, renorm: p.w / keptMass.value }))
);

const droppedPolicies = computed(() =>
  evaluatedPolicies.value.filter((p) => p.crossed)
);

const statusClass = computed(() => {
  if (isSkipped.value) return 'is-skip';
  return droppedMass.value > 0 ? 'is-wrap' : 'is-valid';
});

function selectPreset(id) {
  presetId.value = id;
  d4Index.value = 0;
  const p = PRESETS.find((item) => item.id === id) ?? PRESETS[0];
  dx.value = p.defaultShift.dx;
  dy.value = p.defaultShift.dy;
  clampShift();
}

function applyRandomD4() {
  d4Index.value = (d4Index.value + 1 + Math.floor(Math.random() * 7)) % 8;
  clampShift();
}

function randomShift() {
  if (isSkipped.value) return;
  const [minX, maxX] = dxRange.value;
  const [minY, maxY] = dyRange.value;
  dx.value = minX + Math.floor(Math.random() * (maxX - minX + 1));
  dy.value = minY + Math.floor(Math.random() * (maxY - minY + 1));
}

function resetView() {
  selectPreset(presetId.value);
}

function cellCenter(i) {
  return i * CELL + CELL / 2;
}

function fmtShift(v) {
  return v > 0 ? `+${v}` : `${v}`;
}

function dxRangeStr([lo, hi]) {
  return `${fmtShift(lo)} .. ${fmtShift(hi)}`;
}
</script>

<style scoped>
*, *::before, *::after {
  box-sizing: border-box;
}

.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  display: flex;
  justify-content: center;
  padding: 18px 14px;
  width: 100%;
  max-width: 100%;
  overflow-x: hidden;
}

.container {
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
}

h1 {
  font-size: 19px;
  font-weight: 700;
  text-align: center;
  margin: 0 0 12px 0;
  color: #0f172a;
  letter-spacing: -0.2px;
}

.controls-panel {
  display: flex;
  flex-direction: column;
  gap: 9px;
  margin-bottom: 10px;
}

.btn-row,
.shift-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.cfg-btn,
.action-btn,
.step-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 5px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.cfg-btn:hover,
.action-btn:hover:not(:disabled),
.step-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.cfg-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.action-btn.primary {
  background: #2563eb;
  border-color: #2563eb;
  color: #ffffff;
}

.action-btn.primary:hover:not(:disabled) {
  background: #1d4ed8;
}

.action-btn:disabled,
.step-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.nudge-group {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12.5px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 3px 7px;
}

.step-btn {
  padding: 2px 8px;
}

.reset-btn {
  padding: 5px 10px;
}

.status-bar {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 9px 13px;
  margin-bottom: 10px;
  font-size: 12.5px;
  line-height: 1.45;
  color: #334155;
}

.status-bar.is-valid {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-wrap {
  background: #fffbeb;
  border-color: #fcd34d;
}

.status-bar.is-skip {
  background: #fef2f2;
  border-color: #fca5a5;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.source-note {
  margin: 8px 4px 0 4px;
  font-size: 12px;
  line-height: 1.45;
  color: #64748b;
  text-align: center;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
