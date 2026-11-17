<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Minimax Request-Handling Pipeline</h1>
      <p class="hint">Pick a difficulty to follow its path through the engine.</p>

      <div class="picker" role="tablist" aria-label="Difficulty">
        <button
          v-for="m in MODES"
          :key="m.id"
          type="button"
          role="tab"
          :aria-selected="mode === m.id"
          :class="['pick', m.id, { active: mode === m.id }]"
          @click="mode = m.id"
        >
          {{ m.id }}
        </button>
      </div>

      <ol class="flow">
        <li class="stage request">
          <p class="stage-step">1 · Request</p>
          <p class="stage-title">WebSocket JSON</p>
          <ul>
            <li>full <code>board</code> every time</li>
            <li><code>nextPlayer</code>, <code>lastPlay</code></li>
            <li><code>goal</code>, capture and double-three flags</li>
            <li><code>difficulty: "{{ mode }}"</code></li>
          </ul>
        </li>

        <li class="arrow" aria-hidden="true" />

        <li class="stage parse">
          <p class="stage-step">2 · Parse</p>
          <p class="stage-title">Build the board</p>
          <ul>
            <li>two bitboards, <code>uint64_t[19]</code> per player</li>
            <li>one bit per cell, one word per row</li>
            <li>Zobrist hash for the transposition table</li>
          </ul>
        </li>

        <li class="arrow" aria-hidden="true" />

        <li :class="['stage', 'search', mode]">
          <p class="stage-step">3 · Search</p>
          <p class="stage-title">{{ current.algo }}</p>
          <dl class="facts">
            <div><dt>Limit</dt><dd>{{ current.limit }}</dd></div>
            <div><dt>Leaf score</dt><dd><code>{{ current.evalFn }}</code></dd></div>
            <div><dt>Quiescence</dt><dd>{{ current.quiescence }}</dd></div>
            <div><dt>Move order</dt><dd>{{ current.ordering }}</dd></div>
          </dl>
          <p class="shared">Shared by all modes: neighbor-mask move generation, one process-wide transposition table, killer moves.</p>
        </li>

        <li class="arrow" aria-hidden="true" />

        <li class="stage response">
          <p class="stage-step">4 · Response</p>
          <p class="stage-title">JSON back to the client</p>
          <ul>
            <li>AI move in <code>lastPlay</code></li>
            <li>updated <code>board</code></li>
            <li><code>capturedStones</code></li>
            <li><code>executionTime</code></li>
          </ul>
        </li>
      </ol>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const MODES = [
  {
    id: 'easy',
    algo: 'Alpha-beta',
    limit: 'Depth 5',
    evalFn: 'evaluatePosition',
    quiescence: 'Yes, capture moves only, when captures are on',
    ordering: 'Hash move, killer moves, then evaluation',
  },
  {
    id: 'medium',
    algo: 'Iterative deepening (alpha-beta)',
    limit: 'Deeper each pass, up to depth 10, stops at 0.4 s',
    evalFn: 'evaluatePosition',
    quiescence: 'Yes, capture moves only, when captures are on',
    ordering: 'Hash move, killer moves, then evaluation',
  },
  {
    id: 'hard',
    algo: 'Principal variation search (PVS)',
    limit: 'Depth 10',
    evalFn: 'evaluatePositionHard',
    quiescence: 'No, returns the static score at depth 0',
    ordering: 'Killer moves, then evaluation',
  },
];

const mode = ref('hard');
const current = computed(() => MODES.find((m) => m.id === mode.value));
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
.container { width: 100%; max-width: 1100px; margin: 0 auto; }

h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 12px; }

.picker { display: flex; justify-content: center; gap: 8px; margin-bottom: 18px; }
.pick {
  font: 600 13px/1 inherit; font-family: inherit;
  padding: 8px 18px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
  text-transform: capitalize; transition: all 0.15s;
}
.pick:hover { border-color: #94a3b8; }
.pick.easy.active { background: #dcfce7; border-color: #16a34a; color: #14532d; }
.pick.medium.active { background: #fef3c7; border-color: #d97706; color: #78350f; }
.pick.hard.active { background: #ede9fe; border-color: #7c3aed; color: #4c1d95; }

.flow {
  list-style: none;
  display: grid;
  grid-template-columns: 1fr 24px 1fr 24px 1.5fr 24px 1fr;
  align-items: stretch;
}

.stage {
  background: #fff; border: 1.5px solid; border-radius: 10px;
  padding: 12px 14px; min-width: 0;
}
.stage.request, .stage.response { border-color: #93c5fd; background: #eff6ff; }
.stage.parse { border-color: #5eead4; background: #f0fdfa; }
.stage.search { border-width: 2px; transition: background 0.2s, border-color 0.2s; }
.stage.search.easy { border-color: #16a34a; background: #f0fdf4; }
.stage.search.medium { border-color: #d97706; background: #fffbeb; }
.stage.search.hard { border-color: #7c3aed; background: #f5f3ff; }

.stage-step { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #64748b; }
.stage-title { font-size: 15px; font-weight: 700; color: #0f172a; margin: 2px 0 8px; }
.stage ul { padding-left: 16px; }
.stage li { font-size: 13px; line-height: 1.55; list-style: disc; }

.facts { display: flex; flex-direction: column; gap: 6px; }
.facts div { display: grid; grid-template-columns: 88px 1fr; gap: 8px; font-size: 13px; line-height: 1.45; }
.facts dt { font-weight: 700; color: #475569; }
.shared { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #cbd5e1; font-size: 12px; line-height: 1.5; color: #475569; }

.arrow { position: relative; }
.arrow::before {
  content: ''; position: absolute; top: 50%; left: 3px; right: 7px;
  border-top: 2px solid #94a3b8;
}
.arrow::after {
  content: ''; position: absolute; top: calc(50% - 5px); right: 1px;
  border: 6px solid transparent; border-left: 7px solid #94a3b8; border-right: 0;
}

code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em; background: rgba(15, 23, 42, 0.06); padding: 0.05em 0.3em; border-radius: 4px;
  overflow-wrap: anywhere;
  font-variant-ligatures: none;
}

@container (max-width: 900px) {
  .flow { grid-template-columns: 1fr; }
  .stage { display: grid; grid-template-columns: 150px 1fr; column-gap: 16px; align-items: start; }
  .stage-step, .stage-title { grid-column: 1; }
  .stage > ul, .stage > .facts, .stage > .shared { grid-column: 2; grid-row: 1 / span 3; }
  .stage > .shared { grid-row: auto; }
  .arrow { height: 22px; }
  .arrow::before { top: 2px; bottom: 8px; left: 50%; right: auto; border-top: 0; border-left: 2px solid #94a3b8; }
  .arrow::after { top: auto; bottom: 1px; left: calc(50% - 5px); right: auto; border: 6px solid transparent; border-top: 7px solid #94a3b8; border-bottom: 0; }
}

@container (max-width: 520px) {
  .stage { display: block; }
  .facts div { grid-template-columns: 80px 1fr; }
}
</style>
