<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Pruning Dead Channels Inside a Residual Block</h1>

      <div class="controls-panel">
        <div class="channel-row">
          <span class="row-label">Toggle channel:</span>
          <button
            v-for="ch in 6"
            :key="'btn-ch-' + (ch - 1)"
            type="button"
            class="ch-btn"
            :class="{ dead: deadFlags[ch - 1], special: conv1ZeroOnly === ch - 1 && !deadFlags[ch - 1] }"
            @click="toggleChannel(ch - 1)"
          >
            c{{ ch - 1 }}: {{ channelStateText(ch - 1) }}
          </button>
        </div>

        <div class="preset-row">
          <button type="button" class="preset-btn" @click="applyPreset('twoDead')">
            2 dead (c1, c4)
          </button>
          <button type="button" class="preset-btn" @click="applyPreset('conv1Only')">
            conv1[c2]=0, conv2 reads (kept)
          </button>
          <button type="button" class="preset-btn" @click="applyPreset('allDead')">
            All dead (c0 guard)
          </button>
          <button type="button" class="preset-btn" @click="applyPreset('allAlive')">
            All alive (6 / 6)
          </button>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 434"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Residual block channel pruning diagram showing which conv1, bn1, and conv2 slices are removed while the skip connection stays unchanged"
      >
        <defs>
          <marker id="d22-arr" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
            <path d="M0,0 L7,3 L0,6 Z" fill="#475569" />
          </marker>
          <marker id="d22-arr-blue" markerWidth="7" markerHeight="6" refX="6" refY="3" orient="auto">
            <path d="M0,0 L7,3 L0,6 Z" fill="#2563eb" />
          </marker>
        </defs>

        <!-- Illustrative banner -->
        <rect x="20" y="8" width="740" height="24" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="24" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Top: illustrative 6-channel ResBlock (click c0–c5 to toggle) · Bottom: measured v5 iter 144 blocks (deploy-v5-pruned.yaml)
        </text>

        <!-- Skip connection path (unchanged) -->
        <path
          d="M 76,170 L 76,66 L 672,66 L 672,152"
          fill="none"
          stroke="#2563eb"
          stroke-width="2.2"
          marker-end="url(#d22-arr-blue)"
        />
        <rect x="250" y="48" width="280" height="24" rx="5" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
        <text x="390" y="64" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">
          Skip connection: 6 ch unchanged (num_hidden)
        </text>

        <!-- Input x -->
        <rect x="20" y="138" width="66" height="64" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6" />
        <text x="53" y="166" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e40af">Input x</text>
        <text x="53" y="184" text-anchor="middle" font-size="10.5" fill="#1d4ed8">6 ch</text>

        <!-- Arrow x -> conv1 -->
        <line x1="86" y1="170" x2="112" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- conv1 block: rows = output mid channels -->
        <rect x="116" y="96" width="124" height="148" rx="8" fill="#ffffff" stroke="#475569" stroke-width="1.5" />
        <text x="178" y="114" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          conv1 (out rows)
        </text>
        <g v-for="r in 6" :key="'c1-' + (r - 1)">
          <rect
            x="128"
            :y="122 + (r - 1) * 18"
            width="100"
            height="14"
            rx="3"
            :fill="rowFill(r - 1)"
            :stroke="rowStroke(r - 1)"
            :stroke-dasharray="isCut(r - 1) ? '3 2' : 'none'"
            stroke-width="1.2"
          />
          <text x="178" :y="133 + (r - 1) * 18" text-anchor="middle" font-size="9.5" font-weight="600" :fill="rowText(r - 1)">
            {{ conv1RowLabel(r - 1) }}
          </text>
        </g>
        <text x="178" y="238" text-anchor="middle" font-size="9.5" fill="#475569">
          + conv1.bias[used]
        </text>

        <!-- Arrow conv1 -> bn1 -->
        <line x1="240" y1="170" x2="262" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- bn1 + ReLU block -->
        <rect x="266" y="96" width="104" height="148" rx="8" fill="#ffffff" stroke="#475569" stroke-width="1.5" />
        <text x="318" y="114" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          bn1 → ReLU
        </text>
        <g v-for="c in 6" :key="'bn1-' + (c - 1)" class="clickable" @click="toggleChannel(c - 1)">
          <rect
            x="278"
            :y="122 + (c - 1) * 18"
            width="80"
            height="14"
            rx="3"
            :fill="isCut(c - 1) ? '#fef2f2' : '#ecfdf5'"
            :stroke="isCut(c - 1) ? '#f87171' : '#10b981'"
            :stroke-dasharray="isCut(c - 1) ? '3 2' : 'none'"
            stroke-width="1.2"
          />
          <text
            x="318"
            :y="133 + (c - 1) * 18"
            text-anchor="middle"
            font-size="9.5"
            font-weight="600"
            :fill="isCut(c - 1) ? '#b91c1c' : '#065f46'"
          >
            c{{ c - 1 }} {{ isCut(c - 1) ? 'cut' : 'kept' }}
          </text>
        </g>
        <text x="318" y="238" text-anchor="middle" font-size="9.5" fill="#475569">
          γ, β, μ, σ² [used]
        </text>

        <!-- Arrow bn1 -> conv2 -->
        <line x1="370" y1="170" x2="392" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- conv2 block: columns = input mid channels -->
        <rect x="396" y="96" width="144" height="148" rx="8" fill="#ffffff" stroke="#475569" stroke-width="1.5" />
        <text x="468" y="114" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          conv2 (in cols)
        </text>
        <g v-for="col in 6" :key="'c2-' + (col - 1)" class="clickable" @click="toggleChannel(col - 1)">
          <rect
            :x="408 + (col - 1) * 20"
            y="124"
            width="16"
            height="94"
            rx="3"
            :fill="isCut(col - 1) ? '#fef2f2' : '#dbeafe'"
            :stroke="isCut(col - 1) ? '#f87171' : '#2563eb'"
            :stroke-dasharray="isCut(col - 1) ? '3 2' : 'none'"
            stroke-width="1.2"
          />
          <text
            :x="416 + (col - 1) * 20"
            y="174"
            text-anchor="middle"
            font-size="9.5"
            font-weight="700"
            :fill="isCut(col - 1) ? '#b91c1c' : '#1e40af'"
          >
            {{ isCut(col - 1) ? '0' : 'c' + (col - 1) }}
          </text>
        </g>
        <text x="468" y="238" text-anchor="middle" font-size="9.5" fill="#475569">
          conv2.bias (6,) untouched
        </text>

        <!-- Arrow conv2 -> bn2 -->
        <line x1="540" y1="170" x2="562" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- bn2 -->
        <rect x="566" y="138" width="66" height="64" rx="7" fill="#f8fafc" stroke="#475569" stroke-width="1.5" />
        <text x="599" y="166" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">bn2</text>
        <text x="599" y="183" text-anchor="middle" font-size="10" fill="#475569">6 ch fixed</text>

        <!-- Arrow bn2 -> add -->
        <line x1="632" y1="170" x2="652" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- Skip add circle -->
        <circle cx="672" cy="170" r="15" fill="#eff6ff" stroke="#2563eb" stroke-width="1.8" />
        <text x="672" y="175" text-anchor="middle" font-size="15" font-weight="700" fill="#1e40af">+</text>

        <!-- Arrow add -> output y -->
        <line x1="687" y1="170" x2="704" y2="170" stroke="#475569" stroke-width="1.8" marker-end="url(#d22-arr)" />

        <!-- Output y -->
        <rect x="708" y="138" width="52" height="64" rx="7" fill="#eff6ff" stroke="#2563eb" stroke-width="1.6" />
        <text x="734" y="165" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">ReLU</text>
        <text x="734" y="183" text-anchor="middle" font-size="10.5" fill="#1d4ed8">y (6 ch)</text>

        <!-- Live tensor shapes strip -->
        <rect x="20" y="256" width="740" height="44" rx="7" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="34" y="275" font-size="11.5" font-weight="700" fill="#0f172a">
          Kept mid_channels: {{ keptCount }} / 6
          <tspan v-if="guardActive" fill="#b45309"> (c0 kept by guard: block needs ≥ 1 channel)</tspan>
          <tspan v-else-if="conv1ZeroOnly !== null && !deadFlags[conv1ZeroOnly]" fill="#7c3aed">
            (c{{ conv1ZeroOnly }} kept because conv2 still reads its bias/BN constant)
          </tspan>
        </text>
        <text x="34" y="292" font-size="11" fill="#334155">
          Pruned shapes: conv1.weight ({{ keptCount }}, 6, 3, 3) · conv1.bias ({{ keptCount }},) · bn1.* ({{ keptCount }},) · conv2.weight (6, {{ keptCount }}, 3, 3) · conv2.bias (6,)
        </text>

        <!-- Measured v5 iteration 144 blocks panel -->
        <rect x="20" y="310" width="740" height="114" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="34" y="330" font-size="12" font-weight="700" fill="#0f172a">
          Measured v5 Iteration 144 (6×96): 440 of 576 Block Channels Kept (136 Dead Removed)
        </text>

        <g v-for="blk in v5Blocks" :key="'v5-' + blk.idx">
          <text
            :x="34 + blk.idx * 119"
            y="350"
            font-size="10.5"
            font-weight="700"
            fill="#334155"
          >
            Block {{ blk.idx }}: {{ blk.kept }}/96
          </text>
          <rect
            :x="34 + blk.idx * 119"
            y="356"
            width="106"
            height="14"
            rx="3"
            fill="#fecaca"
            stroke="#f87171"
            stroke-width="1"
          />
          <rect
            :x="34 + blk.idx * 119"
            y="356"
            :width="(blk.kept / 96) * 106"
            height="14"
            rx="3"
            fill="#2563eb"
          />
          <text
            :x="34 + blk.idx * 119"
            y="383"
            font-size="9.5"
            fill="#64748b"
          >
            {{ 96 - blk.kept }} dead cut
          </text>
        </g>

        <rect x="34" y="392" width="712" height="24" rx="5" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" />
        <text x="390" y="408" text-anchor="middle" font-size="11" font-weight="600" fill="#1e40af">
          Block multiply-adds: 76% · Output diff on 200 positions: ≤ 5×10⁻⁶ · Oracle A1 (2.5 s, static int8): 613 → 671 searches
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const v5Blocks = [
  { idx: 0, kept: 81 },
  { idx: 1, kept: 77 },
  { idx: 2, kept: 84 },
  { idx: 3, kept: 74 },
  { idx: 4, kept: 65 },
  { idx: 5, kept: 59 },
];

// Channel 1 and 4 start dead in the 6-channel illustrative ResBlock
const deadFlags = ref([false, true, false, false, true, false]);
const conv1ZeroOnly = ref(null);

const guardActive = computed(() => deadFlags.value.every(Boolean));

const usedFlags = computed(() => {
  if (guardActive.value) {
    return [true, false, false, false, false, false];
  }
  return deadFlags.value.map((d) => !d);
});

const keptCount = computed(() => usedFlags.value.filter(Boolean).length);

function isCut(idx) {
  return !usedFlags.value[idx];
}

function toggleChannel(idx) {
  const next = [...deadFlags.value];
  next[idx] = !next[idx];
  deadFlags.value = next;
  if (conv1ZeroOnly.value === idx && next[idx]) {
    conv1ZeroOnly.value = null;
  }
}

function applyPreset(name) {
  if (name === 'twoDead') {
    deadFlags.value = [false, true, false, false, true, false];
    conv1ZeroOnly.value = null;
  } else if (name === 'conv1Only') {
    deadFlags.value = [false, false, false, false, false, false];
    conv1ZeroOnly.value = 2;
  } else if (name === 'allDead') {
    deadFlags.value = [true, true, true, true, true, true];
    conv1ZeroOnly.value = null;
  } else {
    deadFlags.value = [false, false, false, false, false, false];
    conv1ZeroOnly.value = null;
  }
}

function channelStateText(idx) {
  if (guardActive.value && idx === 0) return 'kept (guard)';
  if (deadFlags.value[idx]) return 'dead';
  if (conv1ZeroOnly.value === idx) return 'kept (read)';
  return 'alive';
}

function rowFill(idx) {
  if (isCut(idx)) return '#fef2f2';
  if (conv1ZeroOnly.value === idx) return '#f5f3ff';
  return '#dbeafe';
}

function rowStroke(idx) {
  if (isCut(idx)) return '#f87171';
  if (conv1ZeroOnly.value === idx) return '#7c3aed';
  return '#2563eb';
}

function rowText(idx) {
  if (isCut(idx)) return '#b91c1c';
  if (conv1ZeroOnly.value === idx) return '#5b21b6';
  return '#1e40af';
}

function conv1RowLabel(idx) {
  if (guardActive.value && idx === 0) return 'filter 0 (guard)';
  if (isCut(idx)) return `filter ${idx} (cut)`;
  if (conv1ZeroOnly.value === idx) return `filter ${idx}=0 (kept)`;
  return `filter ${idx} (kept)`;
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
  gap: 8px;
  margin-bottom: 10px;
}

.channel-row,
.preset-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 6px 8px;
}

.row-label {
  font-size: 12px;
  font-weight: 600;
  color: #475569;
}

.ch-btn,
.preset-btn {
  font-family: inherit;
  font-size: 11.5px;
  font-weight: 600;
  padding: 4px 9px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.ch-btn {
  background: #eff6ff;
  border-color: #93c5fd;
  color: #1e40af;
}

.ch-btn.dead {
  background: #fef2f2;
  border-color: #fca5a5;
  color: #991b1b;
  text-decoration: line-through;
}

.ch-btn.special {
  background: #f5f3ff;
  border-color: #c4b5fd;
  color: #5b21b6;
}

.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.clickable {
  cursor: pointer;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
