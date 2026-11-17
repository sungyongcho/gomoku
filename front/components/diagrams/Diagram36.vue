<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Double-Three: A Forbidden Move</h1>
      <p class="hint">Pick an example. Hover a line in the list to pick out that three.</p>

      <div class="picker" role="tablist" aria-label="Example">
        <button
          v-for="(ex, i) in EXAMPLES"
          :key="i"
          type="button"
          role="tab"
          :aria-selected="pick === i"
          :class="['pick', { active: pick === i }]"
          @click="pick = i; focus = -1"
        >
          {{ ex.name }}
        </button>
      </div>

      <div class="stage">
        <div class="board">
          <RuleBoard
            :cols="current.cols"
            :rows="current.rows"
            :stones="current.stones"
            :ghost="{ ...current.move, v: 1, forbidden: true }"
            :lines="current.threes.map((t, i) => ({ ...t, dim: focus !== -1 && focus !== i }))"
            :label="`${current.name}: forbidden point marked with a cross`"
          />
        </div>

        <div class="side">
          <p class="text">
            Black may <strong>not</strong> play on the <span class="x">✕</span>.
            That one stone would make <strong>two free threes at once</strong>, and the opponent can block only one of them.
          </p>
          <ul class="threes">
            <li
              v-for="(t, i) in current.threes"
              :key="i"
              :style="{ '--c': t.color }"
              :class="{ on: focus === i }"
              @mouseenter="focus = i"
              @mouseleave="focus = -1"
              @focus="focus = i"
              @blur="focus = -1"
              tabindex="0"
            >
              <span class="swatch" />
              <span><strong>{{ t.title }}</strong> {{ t.text }}</span>
            </li>
          </ul>
          <p class="note">{{ current.note }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

const BLUE = '#2563eb';
const PURPLE = '#9333ea';

const EXAMPLES = [
  {
    name: 'Example 1',
    cols: 5,
    rows: 5,
    stones: { '1,1': 1, '3,1': 1, '2,2': 1, '3,2': 1 },
    move: { c: 3, r: 3 },
    threes: [
      { from: [3, 1], to: [3, 3], color: BLUE, title: 'Vertical:', text: 'three in a row, both ends empty.' },
      { from: [1, 1], to: [3, 3], color: PURPLE, title: 'Diagonal:', text: 'three in a row, both ends empty.' },
    ],
    note: 'Both threes are open at each end, so either could grow into an open four next turn.',
  },
  {
    name: 'Example 2',
    cols: 5,
    rows: 6,
    stones: { '3,1': 1, '1,2': 1, '3,2': 1, '2,3': 1 },
    move: { c: 3, r: 4 },
    threes: [
      { from: [3, 1], to: [3, 4], color: BLUE, title: 'Vertical:', text: 'three with one gap (● ● · ●), both ends empty.' },
      { from: [1, 2], to: [3, 4], color: PURPLE, title: 'Diagonal:', text: 'three in a row, both ends empty.' },
    ],
    note: 'A three with a single gap still counts: filling the gap makes an open four.',
  },
];

const pick = ref(0);
const focus = ref(-1);
const current = computed(() => EXAMPLES[pick.value]);
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
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 12px; }

.picker { display: flex; justify-content: center; gap: 8px; margin-bottom: 16px; }
.pick {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 8px 18px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155; transition: all 0.15s;
}
.pick:hover { border-color: #94a3b8; }
.pick.active { background: #1e293b; border-color: #1e293b; color: #fff; }

.stage { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: 20px; align-items: center; }
.board { max-width: 300px; width: 100%; justify-self: center; }

.side { display: flex; flex-direction: column; gap: 12px; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }
.x { color: #dc2626; font-weight: 700; }
.threes { list-style: none; display: flex; flex-direction: column; gap: 6px; }
.threes li {
  display: flex; gap: 10px; align-items: center;
  font-size: 13.5px; line-height: 1.45; color: #334155;
  padding: 8px 10px; border-radius: 8px; border: 1.5px solid #e2e8f0; background: #fff;
  cursor: default; transition: border-color 0.15s, background 0.15s;
}
.threes li.on { border-color: var(--c); background: color-mix(in srgb, var(--c) 6%, #fff); }
.threes li:focus-visible { outline: 2px solid var(--c); outline-offset: 1px; }
.swatch { flex: none; width: 26px; height: 12px; border-radius: 999px; border: 2.5px solid var(--c); }
.note { font-size: 13px; line-height: 1.55; color: #64748b; }

@container (max-width: 560px) {
  .stage { grid-template-columns: 1fr; gap: 14px; }
  .board { max-width: 260px; }
}
</style>
