<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Residual Tower Compute vs. 2.5 s Arm Search Budget</h1>

      <div class="controls-group">
        <div class="preset-row" role="group" aria-label="Network presets">
          <button
            v-for="preset in PRESETS"
            :key="preset.id"
            type="button"
            class="preset-btn"
            :class="{ active: blocks === preset.blocks && channels === preset.channels }"
            @click="applyPreset(preset)"
          >
            {{ preset.label }}
          </button>
        </div>

        <div class="sliders-grid">
          <div class="slider-item">
            <label for="blocks-slider" class="slider-label">
              Residual blocks <code>B</code>: <strong>{{ blocks }}</strong>
            </label>
            <input
              id="blocks-slider"
              v-model.number="blocks"
              type="range"
              min="2"
              max="16"
              step="1"
              class="range-input"
            />
          </div>

          <div class="slider-item">
            <label for="channels-slider" class="slider-label">
              Hidden channels <code>C</code>: <strong>{{ channels }}</strong>
            </label>
            <input
              id="channels-slider"
              v-model.number="channels"
              type="range"
              min="32"
              max="256"
              step="16"
              class="range-input"
            />
          </div>
        </div>
      </div>

      <div class="status-bar" :class="matchedPreset?.measuredArm ? 'is-measured' : 'is-estimated'">
        <div class="status-main">
          <span class="badge" :class="matchedPreset?.measuredArm ? 'badge-green' : 'badge-amber'">
            {{ matchedPreset?.measuredArm ? 'Measured on Oracle A1' : 'Illustrative 1 / (B·C²) estimate (not measured)' }}
          </span>
          <span class="status-text">
            <strong>{{ activeTitle }}</strong>:
            <code>B·C² = {{ formatInt(bc2) }}</code>
            (<strong>{{ computeRatio.toFixed(2) }}×</strong> v5) ·
            <strong>{{ (paramCount / 1e6).toFixed(2) }}M</strong> weights ·
            <strong>{{ (towerMacs / 1e6).toFixed(1) }}M</strong> tower MACs
          </span>
        </div>
        <div class="status-sub">
          <template v-if="matchedPreset?.measuredArm">
            2.5 s on 2 Neoverse-N1 cores (static int8, 2 threads):
            <strong>{{ matchedPreset.measuredArm }} searches</strong>
            <template v-if="matchedPreset.c4dMs">
              · C4D worker self-play (bfloat16, batch 8): <strong>{{ matchedPreset.c4dMs }} ms</strong>
            </template>
            <template v-if="matchedPreset.id === 'v6b'">
              (2.33× fewer searches than v5 for 2.96× tower MACs; tree and head work stay fixed)
            </template>
          </template>
          <template v-else-if="matchedPreset?.id === 'v4'">
            V4 (12×128) was not benchmarked on the A1 Arm instance; naive inverse <code>B·C²</code> scaling gives ~{{ estSearches }} searches in 2.5 s (illustrative).
          </template>
          <template v-else>
            Scaling linearly from v5's measured <code>634</code> searches via <code>634 × 55,296 / (B·C²)</code> gives <strong>~{{ estSearches }} searches</strong> (illustrative only; fixed tree and head overhead flatten real ratios).
          </template>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 345"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Residual tower compute scaling and 2.5-second Arm search budget comparison"
      >
        <!-- Panel 1: Relative tower compute B * C^2 -->
        <text x="24" y="24" font-size="14" font-weight="700" fill="#0f172a">
          1. Residual Tower Multiply-Adds (B · C² relative to v5 6×96 = 1.00×)
        </text>
        <text x="756" y="24" text-anchor="end" font-size="11.5" fill="#64748b">
          Max axis shown: 8.0× (16×256 = 19.0×)
        </text>

        <!-- Reference rows + active row -->
        <g v-for="(item, idx) in computeRows" :key="item.key">
          <rect
            x="16"
            :y="38 + idx * 40"
            width="748"
            height="34"
            rx="6"
            :fill="item.isSelected ? '#eff6ff' : 'transparent'"
            :stroke="item.isSelected ? '#93c5fd' : 'transparent'"
            stroke-width="1.5"
          />
          <text
            x="26"
            :y="59 + idx * 40"
            font-size="12.5"
            :font-weight="item.isSelected ? '700' : '600'"
            :fill="item.isSelected ? '#1e3a8a' : '#334155'"
          >
            {{ item.label }}
          </text>

          <rect x="210" :y="45 + idx * 40" width="410" height="20" rx="4" fill="#f1f5f9" stroke="#e2e8f0" stroke-width="1" />
          <rect
            x="210"
            :y="45 + idx * 40"
            :width="ratioToW(item.ratio)"
            height="20"
            rx="4"
            :fill="item.color"
          />

          <text
            x="630"
            :y="59 + idx * 40"
            font-size="12.5"
            :font-weight="item.isSelected ? '700' : '600'"
            :fill="item.isSelected ? '#0f172a' : '#475569'"
          >
            {{ item.ratio.toFixed(2) }}× ({{ formatInt(item.bc2) }})
          </text>
        </g>

        <!-- Divider -->
        <line x1="24" y1="208" x2="756" y2="208" stroke="#e2e8f0" stroke-width="1.5" />

        <!-- Panel 2: 2.5-second search budget on 2 Neoverse-N1 cores -->
        <text x="24" y="232" font-size="14" font-weight="700" fill="#0f172a">
          2. Searches in 2.5 s on Two Neoverse-N1 Cores (static int8 ONNX, 2 threads)
        </text>

        <g v-for="(row, idx) in searchRows" :key="row.key">
          <rect
            x="16"
            :y="244 + idx * 32"
            width="748"
            height="28"
            rx="5"
            :fill="row.isSelected ? '#eff6ff' : 'transparent'"
            :stroke="row.isSelected ? '#93c5fd' : 'transparent'"
            stroke-width="1.2"
          />
          <text
            x="26"
            :y="262 + idx * 32"
            font-size="12.5"
            :font-weight="row.isSelected ? '700' : '600'"
            :fill="row.isSelected ? '#1e3a8a' : '#334155'"
          >
            {{ row.label }}
          </text>

          <rect x="210" :y="249 + idx * 32" width="410" height="18" rx="4" fill="#f1f5f9" stroke="#e2e8f0" stroke-width="1" />
          <rect
            x="210"
            :y="249 + idx * 32"
            :width="searchesToW(row.searches)"
            height="18"
            rx="4"
            :fill="row.color"
            :stroke="row.isEstimate ? '#b45309' : 'none'"
            :stroke-dasharray="row.isEstimate ? '4 2' : 'none'"
          />

          <text
            x="630"
            :y="262 + idx * 32"
            font-size="12.5"
            :font-weight="row.isSelected ? '700' : '600'"
            :fill="row.isEstimate ? '#92400e' : '#0f172a'"
          >
            {{ row.valueText }}
          </text>
        </g>
      </svg>

      <p class="source-note">
        Sources: <code>PolicyValueNet</code> (<code>policy_value_net.py</code>), <code>v6-10x128.yaml</code>, and <code>deploy/alphazero-oracle/README.md</code> (<code>634</code> vs <code>272</code> measured at 2.5 s). Any configuration other than <code>6×96</code> or <code>10×128</code> shows an illustrative <code>1/(B·C²)</code> search estimate, not a benchmark.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const PRESETS = [
  {
    id: 'v5',
    label: 'v5 (6×96)',
    blocks: 6,
    channels: 96,
    measuredArm: 634,
    c4dMs: 16,
  },
  {
    id: 'v6b',
    label: 'v6 / v6b (10×128)',
    blocks: 10,
    channels: 128,
    measuredArm: 272,
    c4dMs: 39,
  },
  {
    id: 'v4',
    label: 'V4 (12×128)',
    blocks: 12,
    channels: 128,
    measuredArm: null,
    c4dMs: null,
  },
];

const V5_BC2 = 6 * 96 * 96; // 55,296
const V5_SEARCHES = 634;

const blocks = ref(10);
const channels = ref(128);

function applyPreset(preset) {
  blocks.value = preset.blocks;
  channels.value = preset.channels;
}

const matchedPreset = computed(() =>
  PRESETS.find((p) => p.blocks === blocks.value && p.channels === channels.value) ?? null
);

const activeTitle = computed(() =>
  matchedPreset.value ? matchedPreset.value.label : `Custom (${blocks.value}×${channels.value})`
);

const bc2 = computed(() => blocks.value * channels.value * channels.value);
const computeRatio = computed(() => bc2.value / V5_BC2);
const towerMacs = computed(() => 2 * 9 * 361 * bc2.value);

const paramCount = computed(() => {
  const b = blocks.value;
  const c = channels.value;
  const start = 13 * c * 9 + 2 * c;
  const res = b * (2 * c * c * 9 + 2 * c + 4 * c);
  const pol = c * 2 + 4 + (2 * 361 * 361 + 361);
  const val = c * 1 + 2 + (361 * 256 + 256) + (256 + 1);
  return start + res + pol + val;
});

const estSearches = computed(() => {
  if (matchedPreset.value?.measuredArm) return matchedPreset.value.measuredArm;
  return Math.max(1, Math.round((V5_SEARCHES * V5_BC2) / bc2.value));
});

const computeRows = computed(() => {
  const base = [
    {
      key: 'v5',
      label: 'v5 (6×96)',
      bc2: 6 * 96 * 96,
      ratio: 1.0,
      color: '#16a34a',
      isSelected: blocks.value === 6 && channels.value === 96,
    },
    {
      key: 'v6b',
      label: 'v6 / v6b (10×128)',
      bc2: 10 * 128 * 128,
      ratio: (10 * 128 * 128) / V5_BC2,
      color: '#2563eb',
      isSelected: blocks.value === 10 && channels.value === 128,
    },
    {
      key: 'v4',
      label: 'V4 (12×128)',
      bc2: 12 * 128 * 128,
      ratio: (12 * 128 * 128) / V5_BC2,
      color: '#64748b',
      isSelected: blocks.value === 12 && channels.value === 128,
    },
  ];

  const isPreset = matchedPreset.value !== null;
  base.push({
    key: 'selected',
    label: isPreset ? `Selected: ${matchedPreset.value.label}` : `Selected (${blocks.value}×${channels.value})`,
    bc2: bc2.value,
    ratio: computeRatio.value,
    color: isPreset ? '#0284c7' : '#d97706',
    isSelected: !isPreset,
  });
  return base;
});

const searchRows = computed(() => {
  const rows = [
    {
      key: 's-v5',
      label: 'v5 (6×96) · measured',
      searches: 634,
      color: '#16a34a',
      isEstimate: false,
      isSelected: blocks.value === 6 && channels.value === 96,
      valueText: '634 searches',
    },
    {
      key: 's-v6b',
      label: 'v6b (10×128) · measured',
      searches: 272,
      color: '#2563eb',
      isEstimate: false,
      isSelected: blocks.value === 10 && channels.value === 128,
      valueText: '272 searches',
    },
  ];

  if (matchedPreset.value?.measuredArm) {
    rows.push({
      key: 's-sel',
      label: `Selected: ${matchedPreset.value.label}`,
      searches: matchedPreset.value.measuredArm,
      color: matchedPreset.value.id === 'v5' ? '#16a34a' : '#2563eb',
      isEstimate: false,
      isSelected: true,
      valueText: `${matchedPreset.value.measuredArm} (measured)`,
    });
  } else {
    rows.push({
      key: 's-sel',
      label: `Selected (${blocks.value}×${channels.value})`,
      searches: estSearches.value,
      color: '#fef3c7',
      isEstimate: true,
      isSelected: true,
      valueText: `~${formatInt(estSearches.value)} est.`,
    });
  }
  return rows;
});

function ratioToW(r) {
  return Math.min(410, Math.max(4, (r / 8.0) * 410));
}

function searchesToW(n) {
  return Math.min(410, Math.max(4, (n / 1000) * 410));
}

function formatInt(n) {
  return Number(n).toLocaleString('en-US');
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

.controls-group {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 10px;
}

.preset-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.preset-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 12px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.preset-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.sliders-grid {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 14px 24px;
}

.slider-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 210px;
  flex: 1 1 220px;
  max-width: 320px;
}

.slider-label {
  font-size: 13px;
  color: #334155;
  text-align: center;
}

.range-input {
  width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.status-bar {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.45;
  color: #334155;
}

.status-bar.is-measured {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-estimated {
  background: #fffbeb;
  border-color: #fcd34d;
}

.status-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.badge {
  display: inline-block;
  font-size: 11.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
}

.badge-green {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #86efac;
}

.badge-amber {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fcd34d;
}

.status-sub {
  font-size: 12.5px;
  color: #475569;
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
  .slider-item {
    max-width: 100%;
  }
}
</style>
