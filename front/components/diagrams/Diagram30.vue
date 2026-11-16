<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Move Cycle &amp; Background Pondering Timeline</h1>

      <div class="controls-panel">
        <div class="mode-row">
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: ponderOn }"
            @click="ponderOn = true"
          >
            Ponder: ON (<code>reuse_tree + ponder</code>)
          </button>
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: !ponderOn }"
            @click="setPonderOff"
          >
            Ponder: OFF (<code>reuse_tree</code> only)
          </button>
        </div>

        <div class="slider-row">
          <label for="think-slider" class="slider-label">
            Opponent think time: <strong>{{ thinkTime }} s</strong>
          </label>
          <input
            id="think-slider"
            v-model.number="sliderThink"
            type="range"
            min="0"
            max="20"
            step="1"
            class="range-input"
            @input="onSliderInput"
          />
          <div class="preset-btns">
            <button
              v-for="p in PRESETS"
              :key="p.val"
              type="button"
              class="preset-btn"
              :class="{ active: thinkTime === p.val }"
              :disabled="!ponderOn && p.val === 130"
              @click="selectPreset(p.val)"
            >
              {{ p.label }}
            </button>
          </div>
        </div>
      </div>

      <div class="status-bar" :class="measuredCase ? 'is-measured' : 'is-illustrative'">
        <div class="status-top">
          <span class="badge" :class="measuredCase ? 'badge-green' : 'badge-amber'">
            {{ measuredCase ? 'Measured on Oracle A1 (1 full game, --connections 2)' : 'Illustrative timeline (intermediate think time not measured)' }}
          </span>
          <span v-if="measuredCase && measuredCase.kind === 'table'" class="status-headline">
            Behind move: <strong>{{ measuredCase.total.toLocaleString() }}</strong> searches
            ({{ measuredCase.carried.toLocaleString() }} carried over, {{ measuredCase.new.toLocaleString() }} new) ·
            Answer: <strong>{{ measuredCase.mean }} / {{ measuredCase.max }}</strong> (mean / max) ·
            Memory peak: <strong>{{ measuredCase.mem }}</strong>
          </span>
          <span v-else-if="measuredCase && measuredCase.kind === 'long'" class="status-headline">
            Ponder ran <strong>120 s</strong> (<code>ponder_time_limit_s</code>) on both cores for
            <strong>33,623 searches</strong>, then stopped; 2nd connection hit the
            <strong>50,000-node cap</strong> after ~55 s; container memory peaked at <strong>461 MB</strong>.
          </span>
          <span v-else class="status-headline">
            <template v-if="ponderOn">
              Background search runs on both cores during the {{ thinkTime }} s wait (up to 120 s or 50,000 nodes), then stops before Turn 2 starts.
            </template>
            <template v-else>
              With pondering off, both Neoverse-N1 cores stay idle during the {{ thinkTime }} s opponent wait; only the subtree from Turn 1 is kept.
            </template>
          </span>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 320"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Timeline of a move cycle showing answer search, opponent think time with pondering, stop overhead deduction, and next answer search"
      >
        <!-- Swim lane labels -->
        <rect x="16" y="34" width="748" height="62" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1.2" />
        <text x="26" y="60" font-size="12" font-weight="700" fill="#1e3a8a">Event Loop</text>
        <text x="26" y="77" font-size="11" fill="#1e40af">(2.5 s clock)</text>

        <rect x="16" y="106" width="748" height="62" rx="6" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="1.2" />
        <text x="26" y="132" font-size="12" font-weight="700" fill="#166534">Ponder Thread</text>
        <text x="26" y="149" font-size="11" fill="#15803d">(stoppable)</text>

        <!-- Turn 1 Answer Search -->
        <rect x="120" y="42" width="120" height="46" rx="5" fill="#2563eb" />
        <text x="180" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">Turn 1 Search</text>
        <text x="180" y="78" text-anchor="middle" font-size="10.5" fill="#dbeafe">≤ 2.5 s (2 cores)</text>

        <!-- Send answer + AdvanceRoot -->
        <line x1="240" y1="30" x2="240" y2="174" stroke="#0f172a" stroke-width="1.8" stroke-dasharray="4 3" />
        <text x="235" y="22" text-anchor="end" font-size="10.5" font-weight="700" fill="#0f172a">
          Send move &amp; AdvanceRoot
        </text>

        <!-- Wait window (244 to 244 + waitWidth) -->
        <template v-if="waitWidth > 0">
          <!-- Event loop waiting -->
          <rect x="244" y="42" :width="waitWidth" height="46" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2" />
          <text
            v-if="waitWidth >= 85"
            :x="244 + waitWidth / 2"
            y="69"
            text-anchor="middle"
            font-size="11"
            fill="#64748b"
          >
            {{ thinkTime === 130 ? 'Opponent thinks 130 s (compressed)' : `Opponent thinks ${thinkTime} s` }}
          </text>

          <!-- Ponder active bar -->
          <template v-if="ponderOn">
            <rect
              x="244"
              y="114"
              :width="ponderActiveWidth"
              height="46"
              rx="4"
              fill="#16a34a"
            />
            <text
              v-if="ponderActiveWidth >= 75"
              :x="244 + ponderActiveWidth / 2"
              y="134"
              text-anchor="middle"
              font-size="11.5"
              font-weight="700"
              fill="#ffffff"
            >
              {{ thinkTime === 130 ? 'Ponder: 120 s (33,623 searches)' : `Ponder: ${thinkTime} s` }}
            </text>
            <text
              v-if="ponderActiveWidth >= 105"
              :x="244 + ponderActiveWidth / 2"
              y="150"
              text-anchor="middle"
              font-size="10"
              fill="#dcfce7"
            >
              stoppable=true · 2 cores
            </text>

            <!-- Idle after 120 s cap in 130 s case -->
            <rect
              v-if="thinkTime === 130"
              :x="244 + ponderActiveWidth"
              y="114"
              :width="waitWidth - ponderActiveWidth"
              height="46"
              rx="4"
              fill="#fef3c7"
              stroke="#f59e0b"
              stroke-width="1.2"
            />
            <text
              v-if="thinkTime === 130"
              :x="244 + ponderActiveWidth + (waitWidth - ponderActiveWidth) / 2"
              y="135"
              text-anchor="middle"
              font-size="10"
              font-weight="700"
              fill="#92400e"
            >
              Idle
            </text>
            <text
              v-if="thinkTime === 130"
              :x="244 + ponderActiveWidth + (waitWidth - ponderActiveWidth) / 2"
              y="149"
              text-anchor="middle"
              font-size="9.5"
              fill="#b45309"
            >
              >120 s
            </text>
          </template>
          <template v-else>
            <rect x="244" y="114" :width="waitWidth" height="46" rx="4" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4 3" />
            <text
              v-if="waitWidth >= 60"
              :x="244 + waitWidth / 2"
              y="141"
              text-anchor="middle"
              font-size="11"
              fill="#64748b"
            >
              {{ waitWidth >= 136 ? 'Ponder OFF (cores idle)' : 'Ponder OFF' }}
            </text>
          </template>
        </template>

        <!-- Next request arrives -->
        <line :x1="nextReqX" y1="30" :x2="nextReqX" y2="174" stroke="#dc2626" stroke-width="2" />
        <text :x="nextReqX + 5" y="22" text-anchor="start" font-size="10.5" font-weight="700" fill="#dc2626">
          Request arrives (start_ns)
        </text>

        <!-- Stop + AlignRoot sliver -->
        <rect :x="nextReqX" y="42" :width="stopWidth" height="46" rx="3" fill="#f59e0b" />
        <text
          :x="nextReqX + stopWidth / 2"
          y="192"
          text-anchor="middle"
          font-size="10.5"
          font-weight="700"
          fill="#b45309"
        >
          {{ ponderOn ? 'Stop + Align' : 'AlignRoot' }}
        </text>

        <!-- Turn 2 Answer Search -->
        <rect :x="turn2X" y="42" :width="turn2Width" height="46" rx="5" fill="#2563eb" />
        <text :x="turn2X + turn2Width / 2" y="62" text-anchor="middle" font-size="12" font-weight="700" fill="#ffffff">
          Turn 2 Search
        </text>
        <text :x="turn2X + turn2Width / 2" y="78" text-anchor="middle" font-size="10.5" fill="#dbeafe">
          T_eff = max(2.5 − t_spent, 0.001)
        </text>

        <!-- 3.0 s Target bracket under Turn 2 -->
        <line :x1="nextReqX" y1="98" :x2="nextReqX" y2="104" stroke="#475569" stroke-width="1.5" />
        <line :x1="turn2X + turn2Width" y1="98" :x2="turn2X + turn2Width" y2="104" stroke="#475569" stroke-width="1.5" />
        <line :x1="nextReqX" y1="104" :x2="turn2X + turn2Width" y2="104" stroke="#475569" stroke-width="1.5" />

        <!-- Bottom explanation box -->
        <rect x="16" y="206" width="748" height="100" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="390" y="229" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
          Why Stopping Ponder Does Not Delay the Answer
        </text>
        <text x="390" y="251" text-anchor="middle" font-size="12" fill="#334155">
          1. start_ns is captured before engine.stop_pondering() calls request_stop() and thread.join().
        </text>
        <text x="390" y="271" text-anchor="middle" font-size="12" fill="#334155">
          2. Time spent waiting for the in-flight leaf evaluation + AlignRoot is subtracted from time_limit_s = 2.5 s.
        </text>
        <text x="390" y="291" text-anchor="middle" font-size="12" font-weight="600" fill="#15803d">
          3. Turn 2 runs on both cores alone: early-move new searches stayed 683–707 (vs 681–721 without ponder).
        </text>
      </svg>

      <p class="source-note">
        Sources: <code>alphazero/server/session.py</code>, <code>alphazero/server/websocket.py</code>, <code>alphazero/configs/deploy-v5-pruned.yaml</code>, and <code>deploy/alphazero-oracle/README.md</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const PRESETS = [
  { val: 0, label: '0 s (instant)' },
  { val: 5, label: '5 s (measured)' },
  { val: 15, label: '15 s (measured)' },
  { val: 130, label: '130 s (cap check)' },
];

const ponderOn = ref(true);
const thinkTime = ref(5);
const sliderThink = ref(5);

function onSliderInput() {
  thinkTime.value = sliderThink.value;
}

function selectPreset(val) {
  thinkTime.value = val;
  sliderThink.value = Math.min(20, val);
}

function setPonderOff() {
  ponderOn.value = false;
  if (thinkTime.value === 130) {
    thinkTime.value = 0;
    sliderThink.value = 0;
  }
}

const measuredCase = computed(() => {
  if (!ponderOn.value && thinkTime.value === 0) {
    return {
      kind: 'table',
      total: 1468,
      carried: 646,
      new: 812,
      mean: '2.42 s',
      max: '2.53 s',
      mem: '262 MB',
    };
  }
  if (ponderOn.value && thinkTime.value === 0) {
    return {
      kind: 'table',
      total: 1468,
      carried: 651,
      new: 799,
      mean: '2.42 s',
      max: '2.56 s',
      mem: '263 MB',
    };
  }
  if (ponderOn.value && thinkTime.value === 5) {
    return {
      kind: 'table',
      total: 6363,
      carried: 5303,
      new: 1044,
      mean: '2.51 s',
      max: '2.53 s',
      mem: '307 MB',
    };
  }
  if (ponderOn.value && thinkTime.value === 15) {
    return {
      kind: 'table',
      total: 13859,
      carried: 12484,
      new: 977,
      mean: '2.51 s',
      max: '2.51 s',
      mem: '415 MB',
    };
  }
  if (ponderOn.value && thinkTime.value === 130) {
    return { kind: 'long' };
  }
  return null;
});

const waitWidth = computed(() => {
  if (thinkTime.value === 0) return 0;
  if (thinkTime.value === 130) return 290;
  return Math.round(45 + (thinkTime.value / 20) * 235);
});

const ponderActiveWidth = computed(() => {
  if (thinkTime.value === 130) return Math.round((120 / 130) * waitWidth.value);
  return waitWidth.value;
});

const nextReqX = computed(() => 244 + waitWidth.value + (waitWidth.value > 0 ? 4 : 0));
const stopWidth = computed(() => (ponderOn.value ? 18 : 10));
const turn2X = computed(() => nextReqX.value + stopWidth.value + 2);
const turn2Width = computed(() => Math.max(150, 748 - turn2X.value));
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
  gap: 10px;
  margin-bottom: 10px;
}

.mode-row,
.slider-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.slider-label {
  font-size: 13px;
  color: #334155;
}

.range-input {
  width: 190px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.preset-btns {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 6px;
}

.ctrl-btn,
.preset-btn {
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

.ctrl-btn:hover:not(:disabled),
.preset-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn:disabled,
.preset-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.ctrl-btn.active,
.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
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

.status-bar.is-illustrative {
  background: #fffbeb;
  border-color: #fcd34d;
}

.status-top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
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
  .range-input {
    width: 100%;
  }
}
</style>
