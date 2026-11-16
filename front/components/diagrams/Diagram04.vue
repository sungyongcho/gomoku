<template>
  <div class="diagram-wrapper" tabindex="0" @keydown="onKeydown">
    <div class="container">
      <h1>Pipelined Async MCTS: CPU Search Loop and Ray Inference Actor</h1>

      <div class="controls">
        <button type="button" :disabled="step === 0" @click="goTo(step - 1)">◀ Prev</button>
        <span class="step-label">Step {{ step + 1 }} / {{ STEPS.length }}</span>
        <button type="button" :disabled="step === STEPS.length - 1" @click="goTo(step + 1)">Next ▶</button>
        <button type="button" :class="{ playing: autoTimer }" @click="toggleAuto">
          {{ autoTimer ? '⏸ Pause' : '▶ Auto Play' }}
        </button>
        <button type="button" @click="reset">↺ Reset</button>
      </div>

      <div class="caption">
        <p class="caption-title">{{ current.title }}</p>
        <p class="caption-body" v-html="current.body" />
      </div>

      <svg
        class="timeline"
        viewBox="0 0 1000 290"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        :aria-label="current.title"
      >
        <defs>
          <marker id="d04-arr-disp" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed" />
          </marker>
          <marker id="d04-arr-res" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#059669" />
          </marker>
          <pattern id="d04-hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="8" height="8" fill="#f1f5f9" />
            <line x1="0" y1="0" x2="0" y2="8" stroke="#cbd5e1" stroke-width="3" />
          </pattern>
        </defs>

        <!-- Lane labels -->
        <text x="58" :y="CPU_Y - 4" text-anchor="middle" class="lane-title" fill="#1d4ed8">CPU</text>
        <text x="58" :y="CPU_Y + 13" text-anchor="middle" class="lane-sub">search loop</text>
        <text x="58" :y="GPU_Y - 4" text-anchor="middle" class="lane-title" fill="#c2410c">GPU</text>
        <text x="58" :y="GPU_Y + 13" text-anchor="middle" class="lane-sub">Ray actor</text>

        <!-- Lane backgrounds -->
        <rect :x="TL - 6" :y="CPU_Y - LANE_H / 2" :width="TR - TL + 12" :height="LANE_H" rx="8" fill="#f1f5f9" stroke="#e2e8f0" />
        <rect :x="TL - 6" :y="GPU_Y - LANE_H / 2" :width="TR - TL + 12" :height="LANE_H" rx="8" fill="#fffbeb" stroke="#fde68a" />

        <!-- Overlap / cycle brackets -->
        <g v-for="br in current.brackets" :key="br.label">
          <line :x1="tx(br.from)" :y1="CPU_Y - LANE_H / 2 - 6" :x2="tx(br.from)" :y2="CPU_Y - LANE_H / 2 - 14" stroke="#0891b2" stroke-width="1.5" />
          <line :x1="tx(br.to)" :y1="CPU_Y - LANE_H / 2 - 6" :x2="tx(br.to)" :y2="CPU_Y - LANE_H / 2 - 14" stroke="#0891b2" stroke-width="1.5" />
          <line :x1="tx(br.from)" :y1="CPU_Y - LANE_H / 2 - 14" :x2="tx(br.to)" :y2="CPU_Y - LANE_H / 2 - 14" stroke="#0891b2" stroke-width="1.5" />
          <text :x="(tx(br.from) + tx(br.to)) / 2" :y="CPU_Y - LANE_H / 2 - 20" text-anchor="middle" class="bracket-label">
            {{ br.label }}
          </text>
          <line :x1="tx(br.from)" :y1="CPU_Y - LANE_H / 2 - 6" :x2="tx(br.from)" :y2="GPU_Y + LANE_H / 2" stroke="#0891b2" stroke-width="1" stroke-dasharray="4 3" />
          <line :x1="tx(br.to)" :y1="CPU_Y - LANE_H / 2 - 6" :x2="tx(br.to)" :y2="GPU_Y + LANE_H / 2" stroke="#0891b2" stroke-width="1" stroke-dasharray="4 3" />
        </g>

        <!-- Arrows: dispatch (CPU → actor) and result (actor → drain) -->
        <path
          v-for="a in current.arrows"
          :key="a.key"
          :d="`M${tx(a.from)},${a.up ? GPU_Y - LANE_H / 2 + 2 : CPU_Y + LANE_H / 2 - 2} L${tx(a.to)},${a.up ? CPU_Y + LANE_H / 2 + 4 : GPU_Y - LANE_H / 2 - 4}`"
          fill="none"
          :stroke="a.up ? '#059669' : '#7c3aed'"
          stroke-width="1.6"
          :stroke-dasharray="a.dashed ? '5 3' : 'none'"
          :marker-end="a.up ? 'url(#d04-arr-res)' : 'url(#d04-arr-disp)'"
        />

        <!-- Blocks -->
        <g v-for="b in current.blocks" :key="b.key">
          <rect
            :x="tx(b.from) + 1"
            :y="laneY(b.lane) - BLOCK_H / 2"
            :width="Math.max(tx(b.to) - tx(b.from) - 2, 4)"
            :height="BLOCK_H"
            rx="6"
            :fill="b.kind === 'wait' ? 'url(#d04-hatch)' : PAL[b.kind].f"
            :stroke="PAL[b.kind].s"
            :stroke-width="b.isNew ? 2.6 : 1.4"
            :stroke-dasharray="b.kind === 'wait' ? '4 3' : 'none'"
          />
          <text
            :x="(tx(b.from) + tx(b.to)) / 2"
            :y="laneY(b.lane) + (b.sub ? -3 : 5)"
            text-anchor="middle"
            class="block-label"
            :fill="PAL[b.kind].t"
          >
            {{ b.label }}
          </text>
          <text
            v-if="b.sub"
            :x="(tx(b.from) + tx(b.to)) / 2"
            :y="laneY(b.lane) + 13"
            text-anchor="middle"
            class="block-sub"
            :fill="PAL[b.kind].t"
          >
            {{ b.sub }}
          </text>
        </g>

        <!-- Time axis -->
        <line :x1="TL" :y1="AXIS_Y" :x2="TR" :y2="AXIS_Y" stroke="#94a3b8" stroke-width="1.4" />
        <g v-for="t in [0, 2, 4, 6, 8, 10, 12]" :key="'tick-' + t">
          <line :x1="tx(t)" :y1="AXIS_Y - 4" :x2="tx(t)" :y2="AXIS_Y + 4" stroke="#94a3b8" />
          <text :x="tx(t)" :y="AXIS_Y + 18" text-anchor="middle" class="axis-label">{{ t }}</text>
        </g>
        <text :x="TR" :y="AXIS_Y + 34" text-anchor="end" class="axis-label">time (illustrative units)</text>
      </svg>

      <div class="legend">
        <span><i class="sw sel" /> Select (UCB descent, +VL, enqueue)</span>
        <span><i class="sw disp" /> D = dispatch batch</span>
        <span><i class="sw drain" /> Drain (expand, backup, −VL)</span>
        <span><i class="sw gpu" /> Batch forward pass</span>
        <span><i class="sw wait" /> Waiting on <code>ray.wait</code></span>
      </div>

      <div v-if="current.notes.length" class="notes">
        <div v-for="n in current.notes" :key="n.title" class="note" :class="n.tone">
          <p class="note-title">{{ n.title }}</p>
          <ul>
            <li v-for="line in n.lines" :key="line" v-html="line" />
          </ul>
        </div>
      </div>

      <p class="source-note">
        Illustrative timing with <code>async_inflight_limit = 2</code>. Loop structure from
        <code>RayAsyncEngine.search</code> and <code>BatchInferenceManager</code>
        (<code>alphazero/gomoku/pvmcts/search/ray/</code>).
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue';

// Layout (viewBox units)
const TL = 120;
const TR = 975;
const U = (TR - TL) / 12;
const CPU_Y = 82;
const GPU_Y = 192;
const LANE_H = 56;
const BLOCK_H = 44;
const AXIS_Y = 246;

const tx = (t) => TL + t * U;
const laneY = (lane) => (lane === 'cpu' ? CPU_Y : GPU_Y);

const PAL = {
  sel: { f: '#dbeafe', s: '#2563eb', t: '#1e3a8a' },
  disp: { f: '#c4b5fd', s: '#7c3aed', t: '#ffffff' },
  drain: { f: '#a7f3d0', s: '#059669', t: '#064e3b' },
  gpu: { f: '#fed7aa', s: '#ea580c', t: '#7c2d12' },
  block: { f: '#e2e8f0', s: '#94a3b8', t: '#475569' },
  idle: { f: '#fee2e2', s: '#dc2626', t: '#991b1b' },
  wait: { f: '#f1f5f9', s: '#94a3b8', t: '#475569' },
};

// Stop-and-wait baseline: one batch at a time, each side idle while the other works.
const NAIVE_BLOCKS = [
  { key: 'n-s1', lane: 'cpu', from: 0, to: 1.7, kind: 'sel', label: 'Select B1', sub: '+VL on path' },
  { key: 'n-b1', lane: 'cpu', from: 1.7, to: 4.7, kind: 'block', label: 'Blocked', sub: 'waiting for B1' },
  { key: 'n-d1', lane: 'cpu', from: 4.7, to: 5.7, kind: 'drain', label: 'Drain B1', sub: '−VL' },
  { key: 'n-s2', lane: 'cpu', from: 5.7, to: 7.4, kind: 'sel', label: 'Select B2', sub: '+VL on path' },
  { key: 'n-b2', lane: 'cpu', from: 7.4, to: 10.4, kind: 'block', label: 'Blocked', sub: 'waiting for B2' },
  { key: 'n-d2', lane: 'cpu', from: 10.4, to: 11.4, kind: 'drain', label: 'Drain B2', sub: '' },
  { key: 'n-i0', lane: 'gpu', from: 0, to: 1.7, kind: 'idle', label: 'Idle', sub: '' },
  { key: 'n-g1', lane: 'gpu', from: 1.7, to: 4.7, kind: 'gpu', label: 'Infer B1', sub: 'forward pass' },
  { key: 'n-i1', lane: 'gpu', from: 4.7, to: 7.4, kind: 'idle', label: 'Idle', sub: '' },
  { key: 'n-g2', lane: 'gpu', from: 7.4, to: 10.4, kind: 'gpu', label: 'Infer B2', sub: 'forward pass' },
  { key: 'n-i2', lane: 'gpu', from: 10.4, to: 12, kind: 'idle', label: 'Idle', sub: '' },
];

// Pipelined timeline. `step` is the first step a block appears in.
const PIPE_BLOCKS = [
  { key: 's1', step: 1, lane: 'cpu', from: 0, to: 1.7, kind: 'sel', label: 'Select B1', sub: '+VL on path' },
  { key: 'd1', step: 2, lane: 'cpu', from: 1.7, to: 2.0, kind: 'disp', label: 'D', sub: '' },
  { key: 's2', step: 2, lane: 'cpu', from: 2.0, to: 3.7, kind: 'sel', label: 'Select B2', sub: '+VL on path' },
  { key: 'g1', step: 2, lane: 'gpu', from: 2.0, to: 5.0, kind: 'gpu', label: 'Infer B1', sub: 'forward pass' },
  { key: 'd2', step: 3, lane: 'cpu', from: 3.7, to: 4.0, kind: 'disp', label: 'D', sub: '' },
  { key: 'w1', step: 3, lane: 'cpu', from: 4.0, to: 5.0, kind: 'wait', label: 'Wait', sub: 'pending full' },
  { key: 'r1', step: 4, lane: 'cpu', from: 5.0, to: 6.0, kind: 'drain', label: 'Drain B1', sub: '−VL' },
  { key: 's3', step: 4, lane: 'cpu', from: 6.0, to: 7.7, kind: 'sel', label: 'Select B3', sub: '+VL on path' },
  { key: 'g2', step: 4, lane: 'gpu', from: 5.0, to: 8.0, kind: 'gpu', label: 'Infer B2', sub: 'forward pass' },
  { key: 'd3', step: 6, lane: 'cpu', from: 7.7, to: 8.0, kind: 'disp', label: 'D', sub: '' },
  { key: 'r2', step: 6, lane: 'cpu', from: 8.0, to: 9.0, kind: 'drain', label: 'Drain B2', sub: '−VL' },
  { key: 's4', step: 6, lane: 'cpu', from: 9.0, to: 10.7, kind: 'sel', label: 'Select B4', sub: '+VL on path' },
  { key: 'd4', step: 6, lane: 'cpu', from: 10.7, to: 11.0, kind: 'disp', label: 'D', sub: '' },
  { key: 'r3', step: 6, lane: 'cpu', from: 11.0, to: 12.0, kind: 'drain', label: 'Drain B3', sub: '−VL' },
  { key: 'g3', step: 6, lane: 'gpu', from: 8.0, to: 11.0, kind: 'gpu', label: 'Infer B3', sub: 'forward pass' },
  { key: 'g4', step: 6, lane: 'gpu', from: 11.0, to: 12.0, kind: 'gpu', label: 'Infer B4', sub: '' },
];

const PIPE_ARROWS = [
  { key: 'a-d1', step: 2, from: 1.85, to: 2.05 },
  { key: 'a-d2', step: 3, from: 3.85, to: 5.0, dashed: true },
  { key: 'a-r1', step: 4, from: 5.0, to: 5.0, up: true },
  { key: 'a-d3', step: 6, from: 7.85, to: 8.05 },
  { key: 'a-r2', step: 6, from: 8.0, to: 8.0, up: true },
  { key: 'a-d4', step: 6, from: 10.85, to: 11.05 },
  { key: 'a-r3', step: 6, from: 11.0, to: 11.0, up: true },
];

const STEPS = [
  {
    title: 'Stop-and-wait: one batch at a time',
    body: 'The CPU selects a batch, sends it and <b>blocks</b> until the result returns. The GPU sits idle while the CPU selects and drains; the CPU sits idle while the GPU infers.',
    naive: true,
    brackets: [],
    notes: [],
  },
  {
    title: 'Select B1: UCB descent with virtual loss',
    body: 'For each root the loop walks down by UCB to a leaf, adds <code>pending_visits += 1</code> on every node of the path (+VL), encodes the leaf and enqueues it.',
    brackets: [],
    notes: [],
  },
  {
    title: 'Dispatch B1, keep selecting B2',
    body: 'A batch is sent when the queue reaches <code>batch_size</code> or <code>max_wait_ms</code> has passed. <code>infer_async</code> returns a Ray reference at once, so the loop goes straight back to selecting B2 while the actor runs B1.',
    brackets: [{ from: 2.0, to: 3.7, label: 'Select B2 ‖ Infer B1' }],
    notes: [],
  },
  {
    title: 'Backpressure: pending reaches the limit',
    body: 'B2 is dispatched and waits behind B1 at the actor. Queued plus in-flight requests now equal <code>async_inflight_limit × batch_size</code>, so the selection phase stops and the loop polls <code>ray.wait(num_returns=1, timeout=1 ms)</code>.',
    brackets: [],
    notes: [],
  },
  {
    title: 'Drain B1 as soon as it is ready',
    body: '<code>ray.wait(num_returns=1)</code> returns the first finished batch. Each leaf is expanded from its policy, its value is backed up to the root, and <code>pending_visits -= 1</code> removes the virtual loss. The freed slot lets selection resume (B3) while the actor already runs B2.',
    brackets: [],
    notes: [],
  },
  {
    title: 'Why the virtual loss is needed',
    body: 'B2 and B3 are selected before B1 is backed up. Without a penalty on in-flight paths, every selection would descend to the same promising leaf.',
    brackets: [],
    notes: [
      {
        title: 'Virtual loss and in-flight bookkeeping',
        tone: 'blue',
        lines: [
          '<b>+VL</b>: <code>pending_visits += 1</code> on every node of the selected path makes that path less attractive to the next UCB descent.',
          '<b>−VL</b>: <code>pending_visits -= 1</code> on the same path after the result is backed up.',
          '<code>inflight_to_root</code> skips a leaf that is already waiting for a result.',
          'If the search exits early, every in-flight path has its virtual loss removed in <code>finally</code>.',
        ],
      },
    ],
  },
  {
    title: 'Steady state: drain, select, dispatch',
    body: 'Each loop cycle drains one finished batch, selects the next one and dispatches it. When one cycle of CPU work takes about as long as one forward pass, neither side waits.',
    brackets: [{ from: 8.0, to: 11.0, label: 'one cycle ≈ one forward pass' }],
    notes: [
      {
        title: 'What keeps the pipeline full',
        tone: 'green',
        lines: [
          'Dispatch fires when the queue reaches <code>batch_size</code> or <code>max_wait_ms</code> has elapsed.',
          'At most <code>async_inflight_limit</code> batches are in flight; selection pauses at <code>async_inflight_limit × batch_size</code> pending requests.',
          'A freed in-flight slot triggers <code>dispatch_ready()</code>, so a queued batch leaves immediately.',
        ],
      },
    ],
  },
];

const step = ref(0);
const autoTimer = ref(null);

const current = computed(() => {
  const s = STEPS[step.value];
  if (s.naive) {
    return { ...s, blocks: NAIVE_BLOCKS.map((b) => ({ ...b, isNew: false })), arrows: [] };
  }
  const blocks = PIPE_BLOCKS.filter((b) => b.step <= step.value).map((b) => ({
    ...b,
    isNew: b.step === step.value,
  }));
  const arrows = PIPE_ARROWS.filter((a) => a.step <= step.value);
  return { ...s, blocks, arrows };
});

function goTo(idx) {
  if (idx < 0 || idx >= STEPS.length) return;
  step.value = idx;
}

function stopAuto() {
  if (autoTimer.value) {
    clearInterval(autoTimer.value);
    autoTimer.value = null;
  }
}

function toggleAuto() {
  if (autoTimer.value) {
    stopAuto();
    return;
  }
  if (step.value >= STEPS.length - 1) goTo(0);
  autoTimer.value = setInterval(() => {
    if (step.value >= STEPS.length - 1) {
      stopAuto();
      return;
    }
    goTo(step.value + 1);
  }, 3200);
}

function reset() {
  stopAuto();
  goTo(0);
}

// Arrow keys step through the diagram only while it has focus, so page scrolling keeps working.
function onKeydown(e) {
  if (e.key === 'ArrowRight') {
    e.preventDefault();
    goTo(step.value + 1);
  } else if (e.key === 'ArrowLeft') {
    e.preventDefault();
    goTo(step.value - 1);
  }
}

onBeforeUnmount(stopAuto);
</script>

<style scoped>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  display: flex;
  justify-content: center;
  padding: 24px 16px;
  outline: none;
}
.diagram-wrapper:focus-visible { box-shadow: inset 0 0 0 2px #93c5fd; }
.container { width: 100%; max-width: 1100px; margin: 0 auto; }

h1 {
  font-size: 19px; font-weight: 700; text-align: center;
  margin-bottom: 12px; color: #0f172a; letter-spacing: -0.2px;
}

.controls {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: 8px; margin-bottom: 14px;
}
.controls button {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 6px 14px; border: 1.5px solid #cbd5e1; border-radius: 6px;
  background: #fff; color: #334155; cursor: pointer; transition: all 0.15s;
}
.controls button:hover:not(:disabled) { background: #f1f5f9; border-color: #94a3b8; }
.controls button:disabled { opacity: 0.4; cursor: default; }
.controls button.playing { background: #fef3c7; border-color: #d97706; color: #92400e; }
.step-label { font-size: 13px; font-weight: 600; color: #64748b; min-width: 80px; text-align: center; }

.caption {
  background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; min-height: 76px;
}
.caption-title { font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 3px; }
.caption-body { font-size: 13px; line-height: 1.55; color: #475569; }

.timeline { width: 100%; height: auto; display: block; }
.lane-title { font-size: 15px; font-weight: 800; }
.lane-sub { font-size: 11.5px; font-weight: 600; fill: #64748b; }
.block-label { font-size: 12.5px; font-weight: 700; }
.block-sub { font-size: 10.5px; }
.bracket-label { font-size: 11.5px; font-weight: 700; fill: #0e7490; }
.axis-label { font-size: 11px; fill: #64748b; }

.legend {
  display: flex; flex-wrap: wrap; justify-content: center; gap: 6px 16px;
  font-size: 12px; color: #475569; margin-top: 4px;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { display: inline-block; width: 14px; height: 10px; border-radius: 3px; border: 1.5px solid; }
.sw.sel { background: #dbeafe; border-color: #2563eb; }
.sw.disp { background: #c4b5fd; border-color: #7c3aed; }
.sw.drain { background: #a7f3d0; border-color: #059669; }
.sw.gpu { background: #fed7aa; border-color: #ea580c; }
.sw.wait { background: repeating-linear-gradient(45deg, #f1f5f9 0 3px, #cbd5e1 3px 5px); border-color: #94a3b8; border-style: dashed; }

.notes { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
.note { border-radius: 8px; padding: 10px 14px; border: 1.5px solid; }
.note.blue { background: #eff6ff; border-color: #93c5fd; color: #1e3a8a; }
.note.green { background: #f0fdf4; border-color: #86efac; color: #14532d; }
.note-title { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
.note ul { padding-left: 18px; }
.note li { font-size: 12.5px; line-height: 1.55; list-style: disc; }

code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92em; background: rgba(148, 163, 184, 0.18);
  padding: 0 4px; border-radius: 4px;
}

.source-note { font-size: 11.5px; color: #64748b; text-align: center; margin-top: 10px; line-height: 1.5; }
</style>
