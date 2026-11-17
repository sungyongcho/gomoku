<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>PolicyValueNet Architecture</h1>
      <p class="hint">
        Reference config: 19×19 board, {{ PLANES }} input planes, {{ HIDDEN }} channels, {{ BLOCKS }} residual blocks.
        Click a layer for its shapes and parameter count.
      </p>

      <div class="flow">
        <!-- Input -->
        <section class="card input">
          <div class="card-head">
            <span class="card-title">Input</span>
            <code class="shape strong">{{ shp(PLANES, N, N) }}</code>
          </div>
          <p class="card-note">Encoded board state: {{ PLANES }} planes of {{ N }}×{{ N }} (8 + 5 history moves).</p>
        </section>

        <div class="link" aria-hidden="true"><code class="chip">{{ shp(PLANES, N, N) }}</code></div>

        <!-- Shared trunk -->
        <section class="card trunk">
          <div class="card-head">
            <span class="card-title">Shared trunk</span>
            <span class="card-sub">{{ fmt(TRUNK_PARAMS) }} params</span>
          </div>

          <div class="group">
            <p class="group-title">Start block</p>
            <LayerList :layers="START" />
          </div>

          <div class="link small" aria-hidden="true"><code class="chip">{{ shp(HIDDEN, N, N) }}</code></div>

          <div class="group res">
            <div class="res-head">
              <p class="group-title">Residual block <span class="badge">×{{ BLOCKS }}</span></p>
              <button
                type="button"
                class="pick"
                :class="{ active: expanded }"
                :aria-pressed="expanded"
                @click="toggleExpanded"
              >
                {{ expanded ? 'Collapse layers' : 'Expand layers' }}
              </button>
            </div>
            <p class="res-sub">{{ fmt(RES_PARAMS) }} params each, {{ fmt(RES_PARAMS * BLOCKS) }} for all {{ BLOCKS }}. Shape stays {{ shp(HIDDEN, N, N) }}.</p>

            <div class="res-body">
              <div class="res-rows">
                <LayerList :layers="resMain" />
              </div>
              <div class="skip">
                <span class="skip-label">skip</span>
              </div>
              <div class="res-rows tail">
                <LayerList :layers="RES_TAIL" />
              </div>
            </div>
          </div>
        </section>

        <div class="link" aria-hidden="true"><code class="chip">{{ shp(HIDDEN, N, N) }}</code></div>
        <div class="fork" aria-hidden="true"><span /><span /></div>

        <!-- Heads -->
        <div class="heads">
          <section class="card head policy">
            <div class="card-head">
              <span class="card-title">Policy head</span>
              <span class="card-sub">{{ fmt(POLICY_PARAMS) }} params</span>
            </div>
            <p class="head-in">Reads the trunk output, <code>{{ shp(HIDDEN, N, N) }}</code></p>
            <LayerList :layers="POLICY" />
            <div class="out">
              <div class="out-row">
                <span class="out-name">Policy logits</span>
                <code class="shape strong">B×{{ ACTIONS }}</code>
              </div>
              <p class="out-note">One logit per cell. The net has no mask or softmax: MCTS sets illegal moves to −∞, then applies softmax.</p>
            </div>
          </section>

          <section class="card head value">
            <div class="card-head">
              <span class="card-title">Value head</span>
              <span class="card-sub">{{ fmt(VALUE_PARAMS) }} params</span>
            </div>
            <p class="head-in">Reads the same trunk output, <code>{{ shp(HIDDEN, N, N) }}</code></p>
            <LayerList :layers="VALUE" />
            <div class="out">
              <div class="out-row">
                <span class="out-name">Value</span>
                <code class="shape strong">B×1</code>
              </div>
              <p class="out-note">A score in [−1, 1] for the side to move.</p>
            </div>
          </section>
        </div>
      </div>

      <!-- Totals -->
      <section class="totals">
        <div class="totals-head">
          <span>Total trainable parameters</span>
          <strong>{{ fmt(TOTAL) }}</strong>
        </div>
        <div class="bar" aria-hidden="true">
          <span v-for="p in PARTS" :key="p.id" :class="['seg', p.id]" :style="{ width: pct(p.n) }" />
        </div>
        <ul class="legend">
          <li v-for="p in PARTS" :key="p.id">
            <span :class="['dot', p.id]" />
            <span class="legend-name">{{ p.name }}</span>
            <span class="legend-num">{{ fmt(p.n) }}</span>
            <span class="legend-pct">{{ pct(p.n, 1) }}</span>
          </li>
        </ul>
        <p class="totals-note">Weights and biases only. BatchNorm running mean and variance are buffers, not parameters.</p>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, defineComponent, h, ref } from 'vue';

// Reference config (deploy.yaml / V4): 19x19 board, history_length=5, 128 channels, 12 blocks.
const N = 19;
const ACTIONS = N * N; // game.action_size = row_count * col_count
const PLANES = 8 + 5; // calc_num_planes(history_length=5)
const HIDDEN = 128;
const BLOCKS = 12;
const POLICY_CH = 2;
const VALUE_CH = 1;
const FC = 256; // VALUE_HEAD_FC_UNITS
const K = 3; // start block and residual kernel size

const conv = (cin, cout, k, bias) => cin * cout * k * k + (bias ? cout : 0);
const bn = (c) => 2 * c; // gamma + beta
const linear = (fin, fout) => fin * fout + fout;
const fmt = (n) => n.toLocaleString('en-US');
const shp = (...dims) => ['B', ...dims].join('×');

const SRC = 'policy_value_net.py';

const START = [
  {
    id: 's-conv', name: 'Conv2d 3×3', spec: `${PLANES}→${HIDDEN}, no bias`,
    code: `Conv2d(${PLANES}, ${HIDDEN}, kernel_size=3, padding=1, bias=False)`,
    in: shp(PLANES, N, N), out: shp(HIDDEN, N, N),
    params: conv(PLANES, HIDDEN, K, false), formula: `${PLANES} × ${HIDDEN} × 3 × 3`, src: `${SRC}:106-112`,
  },
  {
    id: 's-bn', name: 'BatchNorm2d', spec: `${HIDDEN}`,
    code: `BatchNorm2d(${HIDDEN})`, in: shp(HIDDEN, N, N), out: shp(HIDDEN, N, N),
    params: bn(HIDDEN), formula: `2 × ${HIDDEN} (scale, shift)`, src: `${SRC}:113`,
  },
  {
    id: 's-relu', name: 'ReLU', spec: '',
    code: 'ReLU()', in: shp(HIDDEN, N, N), out: shp(HIDDEN, N, N),
    params: 0, formula: 'no weights', src: `${SRC}:114`,
  },
];

const RES_CONV = conv(HIDDEN, HIDDEN, K, true);
const RES_LAYERS = [
  {
    id: 'r-conv1', name: 'Conv2d 3×3', spec: `${HIDDEN}→${HIDDEN}, bias`,
    code: `conv1 = Conv2d(${HIDDEN}, ${HIDDEN}, kernel_size=3, padding=1)`,
    params: RES_CONV, formula: `${HIDDEN} × ${HIDDEN} × 3 × 3 + ${HIDDEN}`, src: `${SRC}:156-161`,
  },
  {
    id: 'r-bn1', name: 'BatchNorm2d', spec: `${HIDDEN}`,
    code: `bn1 = BatchNorm2d(${HIDDEN})`, params: bn(HIDDEN), formula: `2 × ${HIDDEN}`, src: `${SRC}:162`,
  },
  {
    id: 'r-relu1', name: 'ReLU', spec: '',
    code: 'fn.relu(self.bn1(self.conv1(x)))', params: 0, formula: 'no weights', src: `${SRC}:174`,
  },
  {
    id: 'r-conv2', name: 'Conv2d 3×3', spec: `${HIDDEN}→${HIDDEN}, bias`,
    code: `conv2 = Conv2d(${HIDDEN}, ${HIDDEN}, kernel_size=3, padding=1)`,
    params: RES_CONV, formula: `${HIDDEN} × ${HIDDEN} × 3 × 3 + ${HIDDEN}`, src: `${SRC}:163-168`,
  },
  {
    id: 'r-bn2', name: 'BatchNorm2d', spec: `${HIDDEN}`,
    code: `bn2 = BatchNorm2d(${HIDDEN})`, params: bn(HIDDEN), formula: `2 × ${HIDDEN}`, src: `${SRC}:169`,
  },
].map((l) => ({ ...l, in: shp(HIDDEN, N, N), out: shp(HIDDEN, N, N), perBlock: true }));

const RES_PARAMS = RES_LAYERS.reduce((s, l) => s + l.params, 0);

const RES_MAIN_COLLAPSED = [
  {
    id: 'r-main', name: 'Conv 3×3 → BN → ReLU → Conv 3×3 → BN', spec: '',
    code: 'x = self.bn2(self.conv2(fn.relu(self.bn1(self.conv1(x)))))',
    in: shp(HIDDEN, N, N), out: shp(HIDDEN, N, N),
    params: RES_PARAMS,
    formula: `2 × (${fmt(RES_CONV)} conv) + 2 × (${bn(HIDDEN)} BN)`,
    src: `${SRC}:174-175`, perBlock: true,
  },
];

const RES_TAIL = [
  {
    id: 'r-add', name: '⊕ Add skip', spec: 'x + F(x)', kind: 'add',
    code: 'x += residual', in: `2 × ${shp(HIDDEN, N, N)}`, out: shp(HIDDEN, N, N),
    params: 0, formula: 'no weights', src: `${SRC}:173, 176`, perBlock: true,
  },
  {
    id: 'r-relu2', name: 'ReLU', spec: '',
    code: 'x = fn.relu(x)', in: shp(HIDDEN, N, N), out: shp(HIDDEN, N, N),
    params: 0, formula: 'no weights', src: `${SRC}:177`, perBlock: true,
  },
];

const POLICY = [
  {
    id: 'p-conv', name: 'Conv2d 1×1', spec: `${HIDDEN}→${POLICY_CH}, no bias`,
    code: `Conv2d(${HIDDEN}, ${POLICY_CH}, kernel_size=1, bias=False)`,
    in: shp(HIDDEN, N, N), out: shp(POLICY_CH, N, N),
    params: conv(HIDDEN, POLICY_CH, 1, false), formula: `${HIDDEN} × ${POLICY_CH}`, src: `${SRC}:122`,
  },
  {
    id: 'p-bn', name: 'BatchNorm2d', spec: `${POLICY_CH}`,
    code: `BatchNorm2d(${POLICY_CH})`, in: shp(POLICY_CH, N, N), out: shp(POLICY_CH, N, N),
    params: bn(POLICY_CH), formula: `2 × ${POLICY_CH}`, src: `${SRC}:123`,
  },
  {
    id: 'p-relu', name: 'ReLU', spec: '',
    code: 'ReLU()', in: shp(POLICY_CH, N, N), out: shp(POLICY_CH, N, N),
    params: 0, formula: 'no weights', src: `${SRC}:124`,
  },
  {
    id: 'p-flat', name: 'Flatten', spec: '',
    code: 'Flatten()', in: shp(POLICY_CH, N, N), out: shp(POLICY_CH * ACTIONS),
    params: 0, formula: `no weights (${POLICY_CH} × ${N} × ${N} = ${POLICY_CH * ACTIONS})`, src: `${SRC}:125`,
  },
  {
    id: 'p-fc', name: 'Linear', spec: `${POLICY_CH * ACTIONS}→${ACTIONS}`,
    code: `Linear(${POLICY_CH * ACTIONS}, ${ACTIONS})`, in: shp(POLICY_CH * ACTIONS), out: shp(ACTIONS),
    params: linear(POLICY_CH * ACTIONS, ACTIONS), formula: `${POLICY_CH * ACTIONS} × ${ACTIONS} + ${ACTIONS}`, src: `${SRC}:126`,
  },
];

const VALUE = [
  {
    id: 'v-conv', name: 'Conv2d 1×1', spec: `${HIDDEN}→${VALUE_CH}, no bias`,
    code: `Conv2d(${HIDDEN}, ${VALUE_CH}, kernel_size=1, bias=False)`,
    in: shp(HIDDEN, N, N), out: shp(VALUE_CH, N, N),
    params: conv(HIDDEN, VALUE_CH, 1, false), formula: `${HIDDEN} × ${VALUE_CH}`, src: `${SRC}:130`,
  },
  {
    id: 'v-bn', name: 'BatchNorm2d', spec: `${VALUE_CH}`,
    code: `BatchNorm2d(${VALUE_CH})`, in: shp(VALUE_CH, N, N), out: shp(VALUE_CH, N, N),
    params: bn(VALUE_CH), formula: `2 × ${VALUE_CH}`, src: `${SRC}:131`,
  },
  {
    id: 'v-relu', name: 'ReLU', spec: '',
    code: 'ReLU()', in: shp(VALUE_CH, N, N), out: shp(VALUE_CH, N, N),
    params: 0, formula: 'no weights', src: `${SRC}:132`,
  },
  {
    id: 'v-flat', name: 'Flatten', spec: '',
    code: 'Flatten()', in: shp(VALUE_CH, N, N), out: shp(VALUE_CH * ACTIONS),
    params: 0, formula: `no weights (${VALUE_CH} × ${N} × ${N} = ${VALUE_CH * ACTIONS})`, src: `${SRC}:133`,
  },
  {
    id: 'v-fc1', name: 'Linear', spec: `${VALUE_CH * ACTIONS}→${FC}`,
    code: `Linear(${VALUE_CH * ACTIONS}, ${FC})`, in: shp(VALUE_CH * ACTIONS), out: shp(FC),
    params: linear(VALUE_CH * ACTIONS, FC), formula: `${VALUE_CH * ACTIONS} × ${FC} + ${FC}`, src: `${SRC}:134`,
  },
  {
    id: 'v-relu2', name: 'ReLU', spec: '',
    code: 'ReLU()', in: shp(FC), out: shp(FC),
    params: 0, formula: 'no weights', src: `${SRC}:135`,
  },
  {
    id: 'v-fc2', name: 'Linear', spec: `${FC}→1`,
    code: `Linear(${FC}, 1)`, in: shp(FC), out: shp(1),
    params: linear(FC, 1), formula: `${FC} × 1 + 1`, src: `${SRC}:136`,
  },
  {
    id: 'v-tanh', name: 'Tanh', spec: '',
    code: 'Tanh()', in: shp(1), out: shp(1),
    params: 0, formula: 'no weights', src: `${SRC}:137`,
  },
];

const sum = (layers) => layers.reduce((s, l) => s + l.params, 0);
const START_PARAMS = sum(START);
const TRUNK_PARAMS = START_PARAMS + RES_PARAMS * BLOCKS;
const POLICY_PARAMS = sum(POLICY);
const VALUE_PARAMS = sum(VALUE);
const TOTAL = TRUNK_PARAMS + POLICY_PARAMS + VALUE_PARAMS; // 3,917,714 (matches PyTorch's count)

const PARTS = [
  { id: 'start', name: 'Start block', n: START_PARAMS },
  { id: 'res', name: `Residual blocks ×${BLOCKS}`, n: RES_PARAMS * BLOCKS },
  { id: 'policy', name: 'Policy head', n: POLICY_PARAMS },
  { id: 'value', name: 'Value head', n: VALUE_PARAMS },
];
const pct = (n, digits) =>
  digits === undefined ? `${(n / TOTAL) * 100}%` : `${((n / TOTAL) * 100).toFixed(digits)}%`;

const expanded = ref(false);
const selected = ref(null);
const resMain = computed(() => (expanded.value ? RES_LAYERS : RES_MAIN_COLLAPSED));

const toggleExpanded = () => {
  expanded.value = !expanded.value;
  if (selected.value && selected.value.startsWith('r-') && selected.value !== 'r-add' && selected.value !== 'r-relu2') {
    selected.value = null;
  }
};
const select = (id) => {
  selected.value = selected.value === id ? null : id;
};

// Renders a list of clickable layers; the selected one opens its details inline.
const LayerList = defineComponent({
  props: { layers: { type: Array, required: true } },
  setup(props) {
    return () =>
      h('ol', { class: 'layers' }, props.layers.map((l) => {
        const open = selected.value === l.id;
        const changes = l.in !== l.out;
        return h('li', { key: l.id, class: ['layer', l.kind, { open }] }, [
          h('button', {
            type: 'button', class: 'layer-btn', 'aria-expanded': open,
            onClick: () => select(l.id),
          }, [
            h('span', { class: 'layer-name' }, [
              l.name,
              l.spec ? h('span', { class: 'layer-spec' }, ` ${l.spec}`) : null,
            ]),
            changes && l.kind !== 'add' ? h('code', { class: 'shape' }, `→ ${l.out}`) : null,
          ]),
          open
            ? h('dl', { class: 'detail' }, [
                h('div', [h('dt', 'Code'), h('dd', [h('code', l.code)])]),
                h('div', [h('dt', 'In'), h('dd', [h('code', l.in)])]),
                h('div', [h('dt', 'Out'), h('dd', [h('code', l.out)])]),
                h('div', [h('dt', 'Params'), h('dd', [
                  h('strong', fmt(l.params)),
                  ` = ${l.formula}`,
                  l.perBlock && l.params ? h('span', { class: 'per-block' }, ` · per block, ×${BLOCKS} = ${fmt(l.params * BLOCKS)}`) : null,
                ])]),
                h('div', [h('dt', 'Source'), h('dd', [h('code', l.src)])]),
              ])
            : null,
        ]);
      }));
  },
});
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
.container { width: 100%; max-width: 820px; margin: 0 auto; }

h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 16px; }

.flow { display: flex; flex-direction: column; align-items: stretch; }

.card {
  background: #fff; border: 1.5px solid; border-radius: 10px;
  padding: 10px 12px; min-width: 0;
}
.card.input { border-color: #93c5fd; background: #eff6ff; max-width: 460px; width: 100%; align-self: center; }
.card.trunk { border-color: #2563eb; background: #eff6ff; max-width: 460px; width: 100%; align-self: center; }
.card.policy { border-color: #16a34a; background: #f0fdf4; }
.card.value { border-color: #7c3aed; background: #f5f3ff; }

.card-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 10px; margin-bottom: 6px; }
.card-title { font-size: 15px; font-weight: 700; color: #0f172a; }
.card-sub { font-size: 12px; color: #475569; }
.card.input .card-head { margin-bottom: 2px; }
.card-note { font-size: 12.5px; line-height: 1.45; color: #475569; }

.group { background: #fff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 8px 10px; }
.group-title { font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #1d4ed8; margin-bottom: 6px; }
.badge {
  display: inline-block; margin-left: 4px; padding: 1px 7px; border-radius: 999px;
  background: #1d4ed8; color: #fff; font-size: 11px; letter-spacing: 0; vertical-align: 1px;
}

.res-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 6px; }
.res-head .group-title { margin-bottom: 0; }
.res-sub { font-size: 12px; line-height: 1.45; color: #475569; margin: 4px 0 8px; }

.pick {
  font-family: inherit; font-size: 12px; font-weight: 600; line-height: 1;
  padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.pick:hover { border-color: #94a3b8; }
.pick.active { background: #dbeafe; border-color: #2563eb; color: #1e3a8a; }

/* Residual body: layer rows on the left, the skip rail on the right. */
.res-body { display: grid; grid-template-columns: minmax(0, 1fr) 30px; column-gap: 4px; }
.res-rows { grid-column: 1; min-width: 0; }
.res-rows.tail { grid-row: 2; margin-top: 4px; }
.skip {
  grid-column: 2; grid-row: 1; position: relative;
  margin-top: 2px; margin-bottom: -20px;
  border: 2px dashed #2563eb; border-left: 0; border-radius: 0 10px 10px 0;
}
.skip::before {
  content: ''; position: absolute; left: -4px; top: -5px; width: 8px; height: 8px;
  border-radius: 50%; background: #2563eb;
}
.skip::after {
  content: ''; position: absolute; left: -9px; bottom: -7px;
  border: 6px solid transparent; border-right: 8px solid #2563eb; border-left: 0;
}
.skip-label {
  position: absolute; top: 50%; right: -2px; transform: translateY(-50%);
  writing-mode: vertical-rl; font-size: 11px; font-weight: 600; color: #1d4ed8;
  background: #fff; padding: 3px 0; white-space: nowrap;
}

/* Layer rows (rendered by LayerList) */
.res-body :deep(.layers), .group :deep(.layers), .card :deep(.layers) { list-style: none; display: flex; flex-direction: column; gap: 4px; }
:deep(.layer-btn) {
  width: 100%; display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 8px;
  font-family: inherit; font-size: 13px; color: #1e293b; text-align: left;
  padding: 6px 9px; border-radius: 6px; cursor: pointer;
  border: 1px solid #e2e8f0; background: #fff;
  transition: border-color 0.15s, background 0.15s;
}
:deep(.layer-btn:hover) { border-color: #94a3b8; }
:deep(.layer-name) { font-weight: 600; min-width: 0; }
:deep(.layer-spec) { font-weight: 400; color: #475569; }
:deep(.shape) { font-size: 11.5px; color: #64748b; white-space: nowrap; }
:deep(.layer.add .layer-btn) { border-style: dashed; border-color: #2563eb; background: #eff6ff; }

.trunk :deep(.layer.open .layer-btn) { border-color: #2563eb; background: #dbeafe; box-shadow: 0 0 0 1px #2563eb; }
.policy :deep(.layer.open .layer-btn) { border-color: #16a34a; background: #dcfce7; box-shadow: 0 0 0 1px #16a34a; }
.value :deep(.layer.open .layer-btn) { border-color: #7c3aed; background: #ede9fe; box-shadow: 0 0 0 1px #7c3aed; }

:deep(.detail) {
  margin: 4px 0 2px; padding: 8px 10px; border-radius: 6px;
  background: #1e293b; color: #e2e8f0;
  display: flex; flex-direction: column; gap: 4px;
}
:deep(.detail div) { display: grid; grid-template-columns: 52px minmax(0, 1fr); gap: 8px; font-size: 12px; line-height: 1.45; }
:deep(.detail dt) { color: #94a3b8; font-weight: 600; }
:deep(.detail dd) { min-width: 0; overflow-wrap: anywhere; }
:deep(.detail code) { font-size: 11.5px; color: #e2e8f0; overflow-wrap: anywhere; }
:deep(.detail strong) { color: #fff; }
:deep(.per-block) { color: #93c5fd; }

code {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none;
}
:deep(code) {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none;
}
.shape.strong { font-size: 12.5px; font-weight: 700; color: #0f172a; }

/* Connectors */
.link { position: relative; display: flex; justify-content: center; height: 34px; align-items: center; }
.link::before {
  content: ''; position: absolute; top: 2px; bottom: 8px; left: 50%;
  border-left: 2px solid #94a3b8;
}
.link::after {
  content: ''; position: absolute; bottom: 1px; left: calc(50% - 6px);
  border: 6px solid transparent; border-top: 7px solid #94a3b8; border-bottom: 0;
}
.link.small { height: 30px; }
.chip {
  position: relative; z-index: 1; font-size: 11.5px; color: #334155;
  background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 999px; padding: 2px 9px;
}
.trunk .chip { background: #eff6ff; }

.fork { position: relative; height: 18px; margin-top: -4px; display: grid; grid-template-columns: 1fr 1fr; column-gap: 16px; }
.fork::before {
  content: ''; position: absolute; top: 0; left: 25%; right: 25%;
  border-top: 2px solid #94a3b8;
}
.fork span { position: relative; }
.fork span::before {
  content: ''; position: absolute; top: 0; bottom: 6px; left: 50%; border-left: 2px solid #94a3b8;
}
.fork span::after {
  content: ''; position: absolute; bottom: 0; left: calc(50% - 5px);
  border: 5px solid transparent; border-top: 6px solid #94a3b8; border-bottom: 0;
}

.head-in { display: none; font-size: 12px; color: #475569; margin: -2px 0 6px; }
.head-in code { font-size: 11.5px; }
.heads { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; align-items: start; }

.out { margin-top: 8px; padding-top: 8px; border-top: 1px dashed; }
.policy .out { border-color: #86efac; }
.value .out { border-color: #c4b5fd; }
.out-row { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 2px 8px; }
.out-name { font-size: 13.5px; font-weight: 700; color: #0f172a; }
.out-note { font-size: 12px; line-height: 1.45; color: #475569; margin-top: 3px; }

/* Totals */
.totals { margin-top: 18px; padding: 12px 14px; background: #fff; border: 1.5px solid #cbd5e1; border-radius: 10px; }
.totals-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 4px 12px; font-size: 14px; font-weight: 600; color: #334155; }
.totals-head strong { font-size: 18px; color: #0f172a; }
.bar { display: flex; height: 10px; border-radius: 999px; overflow: hidden; margin: 10px 0; background: #e2e8f0; }
.seg { display: block; min-width: 2px; }
.seg.start, .dot.start { background: #93c5fd; }
.seg.res, .dot.res { background: #2563eb; }
.seg.policy, .dot.policy { background: #16a34a; }
.seg.value, .dot.value { background: #7c3aed; }
.legend { list-style: none; display: grid; grid-template-columns: 1fr 1fr; gap: 4px 20px; }
.legend li { display: grid; grid-template-columns: 10px minmax(0, 1fr) auto 46px; align-items: center; gap: 8px; font-size: 12.5px; }
.dot { width: 10px; height: 10px; border-radius: 3px; }
.legend-num { font-variant-numeric: tabular-nums; font-weight: 600; }
.legend-pct { font-variant-numeric: tabular-nums; color: #64748b; text-align: right; }
.totals-note { font-size: 12px; line-height: 1.45; color: #64748b; margin-top: 8px; }

@container (max-width: 600px) {
  .heads { grid-template-columns: 1fr; }
  .fork { display: none; }
  .head-in { display: block; }
  .legend { grid-template-columns: 1fr; }
}
</style>
