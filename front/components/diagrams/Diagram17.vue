<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>First-Player Advantage by Random Opening Plies</h1>

      <div class="controls-group">
        <div class="btn-row" role="group" aria-label="Random opening plies">
          <button
            v-for="item in OPENINGS"
            :key="item.plies"
            type="button"
            class="ctrl-btn"
            :class="{ active: selectedPlies === item.plies }"
            @click="selectedPlies = item.plies"
          >
            {{ item.plies }} {{ item.plies === 1 ? 'ply' : 'plies' }}{{ item.plies === 3 ? ' (chosen)' : '' }}
          </button>
        </div>

        <div class="btn-row" role="group" aria-label="Colour balance mode">
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: balanceMode === 'balanced' }"
            @click="balanceMode = 'balanced'"
          >
            Balanced pairs (20B + 20W)
          </button>
          <button
            type="button"
            class="ctrl-btn"
            :class="{ active: balanceMode === 'unbalanced' }"
            @click="balanceMode = 'unbalanced'"
          >
            Unbalanced (+2 Black: 22B + 18W)
          </button>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 395"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Measured Black win rate by number of random opening plies and paired-gate impact"
      >
        <!-- Left/Top panel title -->
        <text x="24" y="24" font-size="13.5" font-weight="700" fill="#0f172a">
          Measured Black Win Rate in Self-Play (Iterations 131–135, 1,200 Games Each)
        </text>
        <text x="756" y="24" text-anchor="end" font-size="11.5" fill="#64748b">
          Click a bar to inspect
        </text>

        <!-- Y-axis gridlines -->
        <g v-for="tick in [0, 25, 50, 75, 100]" :key="'y-' + tick">
          <line
            x1="72"
            :y1="pctToY(tick)"
            x2="758"
            :y2="pctToY(tick)"
            :stroke="tick === 50 ? '#dc2626' : '#e2e8f0'"
            :stroke-width="tick === 50 ? '1.8' : '1'"
            :stroke-dasharray="tick === 50 ? '5 4' : ''"
          />
          <text
            x="64"
            :y="pctToY(tick) + 4"
            text-anchor="end"
            font-size="11.5"
            :font-weight="tick === 50 ? '700' : '500'"
            :fill="tick === 50 ? '#dc2626' : '#64748b'"
          >
            {{ tick }}%
          </text>
        </g>

        <text
          x="754"
          :y="pctToY(50) - 7"
          text-anchor="end"
          font-size="11"
          font-weight="700"
          fill="#dc2626"
        >
          50% balance
        </text>

        <!-- Bars -->
        <g
          v-for="(item, idx) in OPENINGS"
          :key="'bar-' + item.plies"
          class="bar-hit"
          @click="selectedPlies = item.plies"
        >
          <rect
            :x="barX(idx) - 8"
            y="42"
            :width="BAR_W + 16"
            height="194"
            rx="8"
            :fill="selectedPlies === item.plies ? '#eff6ff' : 'transparent'"
            :stroke="selectedPlies === item.plies ? '#93c5fd' : 'transparent'"
            stroke-width="1.5"
          />

          <rect
            :x="barX(idx)"
            :y="pctToY(item.blackWinPct)"
            :width="BAR_W"
            :height="pctToY(0) - pctToY(item.blackWinPct)"
            rx="5"
            :fill="barFill(item)"
            :stroke="item.plies === 3 ? '#15803d' : 'none'"
            :stroke-width="item.plies === 3 ? '2.5' : '0'"
          />

          <text
            :x="barX(idx) + BAR_W / 2"
            :y="pctToY(item.blackWinPct) - 7"
            text-anchor="middle"
            font-size="13"
            font-weight="700"
            fill="#0f172a"
          >
            {{ item.blackWinPct }}%
          </text>

          <text
            :x="barX(idx) + BAR_W / 2"
            y="214"
            text-anchor="middle"
            font-size="12"
            font-weight="700"
            fill="#1e293b"
          >
            {{ item.plies }} {{ item.plies === 1 ? 'ply' : 'plies' }}
          </text>
          <text
            :x="barX(idx) + BAR_W / 2"
            y="229"
            text-anchor="middle"
            font-size="10.5"
            font-weight="600"
            :fill="item.firstMover === 'White' ? '#1d4ed8' : '#475569'"
          >
            {{ item.firstMover }} moves 1st
          </text>
        </g>

        <!-- Bottom Left: Selected opening summary -->
        <rect x="20" y="250" width="358" height="132" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="36" y="273" font-size="13" font-weight="700" fill="#0f172a">
          Selected: {{ active.plies }} Random {{ active.plies === 1 ? 'Ply' : 'Plies' }} ({{ active.firstMover }} Moves First)
        </text>
        <text x="36" y="296" font-size="12" fill="#334155">
          Measured Black win rate: <tspan font-weight="700">{{ active.blackWinPct }}%</tspan>
          (White: <tspan font-weight="700">{{ 100 - active.blackWinPct }}%</tspan>)
        </text>
        <text x="36" y="317" font-size="12" fill="#334155">
          Distance from 50% balance: <tspan font-weight="700">{{ Math.abs(active.blackWinPct - 50) }} percentage points</tspan>
        </text>
        <text x="36" y="340" font-size="11.5" fill="#475569">
          {{ active.noteLine1 }}
        </text>
        <text x="36" y="357" font-size="11.5" fill="#475569">
          {{ active.noteLine2 }}
        </text>

        <!-- Bottom Right: Illustrative 40-game equal-player gate impact -->
        <rect x="394" y="250" width="366" height="132" rx="8" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.5" />
        <text x="410" y="272" font-size="12.5" font-weight="700" fill="#92400e">
          Illustrative: 40-Game Gate Between Equal Players
        </text>
        <text x="410" y="288" font-size="10.5" fill="#92400e">
          Independent games with P(Black wins) = {{ (active.blackWinPct / 100).toFixed(2) }}
        </text>

        <template v-if="balanceMode === 'balanced'">
          <text x="410" y="312" font-size="12" fill="#1e293b">
            20 pairs (20B + 20W): expected score = <tspan font-weight="700">50.0% (20.0 / 40)</tspan>
          </text>
          <text x="410" y="333" font-size="12" fill="#1e293b">
            Pairs split 1–1 by colour: <tspan font-weight="700">{{ pairSplitPct.toFixed(1) }}%</tspan>
            (2–0 or 0–2: {{ pairSweepPct.toFixed(1) }}% each)
          </text>
          <text x="410" y="357" font-size="11" fill="#78350f">
            {{ active.blackWinPct === 82
              ? 'At 82%, over 70% of pairs just trade Black wins (1–1).'
              : active.blackWinPct === 38
                ? 'At 38%, pair splits drop to 52.9% (near the 50% minimum).'
                : 'Stronger colour bias forces more pairs into automatic 1–1 splits.' }}
          </text>
        </template>
        <template v-else>
          <text x="410" y="312" font-size="12" fill="#1e293b">
            22B + 18W: expected challenger score = <tspan font-weight="700">{{ unbalancedScorePct.toFixed(1) }}%</tspan>
          </text>
          <text x="410" y="333" font-size="12" fill="#1e293b">
            Colour shift from 50%: <tspan font-weight="700" :fill="Math.abs(unbalancedShiftPct) >= 2.5 ? '#dc2626' : '#1e293b'">{{ unbalancedShiftPct >= 0 ? '+' : '' }}{{ unbalancedShiftPct.toFixed(1) }} pts</tspan>
            ({{ unbalancedWins.toFixed(1) }} / 40 wins)
          </text>
          <text x="410" y="357" font-size="11" fill="#78350f">
            Just 2 extra Black games push equal models most of the way to 55%.
          </text>
        </template>
      </svg>

      <p class="source-note">
        Measured win rates from <code>v5-small-6x96-cont8.yaml</code> (iterations 131–135, 1,200 games each). Bottom-right probabilities are illustrative under independent-game assumptions.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const OPENINGS = [
  {
    plies: 0,
    blackWinPct: 87,
    firstMover: 'Black',
    noteLine1: 'Empty board: Black places the first searched stone.',
    noteLine2: 'Largest colour skew of all five openings (87% vs 13%).',
  },
  {
    plies: 1,
    blackWinPct: 66,
    firstMover: 'White',
    noteLine1: 'One random Black stone: White makes the first searched move,',
    noteLine2: 'but Black still wins 66% from the extra stone.',
  },
  {
    plies: 2,
    blackWinPct: 79,
    firstMover: 'Black',
    noteLine1: 'Two random stones: Black plays first again in search,',
    noteLine2: 'lifting Black win rate back to 79%.',
  },
  {
    plies: 3,
    blackWinPct: 38,
    firstMover: 'White',
    noteLine1: 'Chosen in cont8 / cont9 / v6 / v6b: closest to 50% balance.',
    noteLine2: 'White moves first in search and wins 62% (Black 38%).',
  },
  {
    plies: 4,
    blackWinPct: 82,
    firstMover: 'Black',
    noteLine1: 'Used in the gate and iter-135 ladder before cont8.',
    noteLine2: 'Iter-135 pairings all scored 9–11/20; gate scored 40–40.',
  },
];

const BAR_W = 78;
const selectedPlies = ref(4);
const balanceMode = ref('balanced');

const active = computed(
  () => OPENINGS.find((o) => o.plies === selectedPlies.value) ?? OPENINGS[4]
);

const pBlack = computed(() => active.value.blackWinPct / 100);
const pairSplitPct = computed(() => {
  const p = pBlack.value;
  return (p * p + (1 - p) * (1 - p)) * 100;
});
const pairSweepPct = computed(() => {
  const p = pBlack.value;
  return p * (1 - p) * 100;
});

const unbalancedWins = computed(() => 22 * pBlack.value + 18 * (1 - pBlack.value));
const unbalancedScorePct = computed(() => (unbalancedWins.value / 40) * 100);
const unbalancedShiftPct = computed(() => unbalancedScorePct.value - 50);

function barX(idx) {
  return 90 + idx * 120;
}

function pctToY(pct) {
  return 196 - (pct / 100) * 144;
}

function barFill(item) {
  if (item.firstMover === 'White') {
    return selectedPlies.value === item.plies ? '#1d4ed8' : '#60a5fa';
  }
  return selectedPlies.value === item.plies ? '#1e293b' : '#64748b';
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
  gap: 8px;
  margin-bottom: 10px;
}

.btn-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.ctrl-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.ctrl-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.bar-hit {
  cursor: pointer;
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
