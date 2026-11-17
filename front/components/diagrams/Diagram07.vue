<template>
  <div class="term-wrap">
    <div class="term">
      <p class="line comment">// Example row: click a column to place or remove a stone</p>

      <div
        v-for="(group, gi) in groups"
        :key="gi"
        :class="['block', group.kind]"
      >
        <div class="line cols">
          <span class="lbl">col:</span>
          <span v-for="c in group.cols" :key="c" class="c">{{ c }}</span>
        </div>
        <div class="line">
          <span class="lbl" />
          <button
            v-for="c in group.cols"
            :key="c"
            type="button"
            :class="['c', 'cell', { on: stones[c], probed: probe === c }]"
            :aria-pressed="stones[c]"
            :aria-label="`Column ${c}`"
            @click="toggle(c)"
            @mouseenter="probe = c"
            @focus="probe = c"
          >{{ stones[c] ? '[●]' : '[ ]' }}</button>
        </div>
        <div class="line">
          <span class="lbl">bit:</span>
          <span
            v-for="c in group.cols"
            :key="c"
            :class="['c', { one: stones[c], probed: probe === c }]"
          >{{ stones[c] ? 1 : 0 }}</span>
        </div>
      </div>

      <p class="line gap"><span class="type">uint64_t</span> row_mask = <span class="num">{{ decimal }}</span>; <span class="comment">// 0x{{ hex }}</span></p>
      <p class="line"><span class="comment">// binary, bit 18 on the left, bit 0 on the right:</span></p>
      <p class="line binary">0b<span v-for="(d, i) in binaryDigits" :key="i" :class="{ one: d === '1', probed: 18 - i === probe, sp: i > 0 && (19 - i) % 4 === 0 }">{{ d }}</span></p>
      <p class="line"><span class="comment">// bits 19–63 are never used and stay 0</span></p>

      <p class="line gap">(row_mask &gt;&gt; <span class="num">{{ probe }}</span>) &amp; 1  ==  <span :class="stones[probe] ? 'one' : 'zero'">{{ stones[probe] ? 1 : 0 }}</span>   <span class="arrow">→</span>  {{ stones[probe] ? 'stone' : 'empty' }} at column {{ probe }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const ALL = Array.from({ length: 19 }, (_, i) => i);
const START = [0, 3, 7, 12, 18];

// One block on wide screens; two blocks (columns 0-9 and 10-18) on narrow ones.
const groups = [
  { kind: 'wide', cols: ALL },
  { kind: 'narrow', cols: ALL.slice(0, 10) },
  { kind: 'narrow', cols: ALL.slice(10) },
];

const stones = ref(ALL.map((c) => START.includes(c)));
const probe = ref(3);

const toggle = (c) => {
  stones.value[c] = !stones.value[c];
  probe.value = c;
};

const value = computed(() => stones.value.reduce((acc, on, c) => (on ? acc + 2 ** c : acc), 0));
const decimal = computed(() => String(value.value));
const hex = computed(() => value.value.toString(16).toUpperCase());
const binaryDigits = computed(() => value.value.toString(2).padStart(19, '0').split(''));
</script>

<style scoped>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.term-wrap { container-type: inline-size; background: #1e293b; }
.term {
  font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-variant-ligatures: none;
  font-size: 14px;
  line-height: 1.7;
  color: #e2e8f0;
  padding: 18px 20px;
  white-space: pre;
}

.line { display: flex; align-items: baseline; min-height: 1.7em; }
p.line { display: block; white-space: pre-wrap; }
.gap { margin-top: 1.1em; }
.block { margin-top: 0.9em; }
.block.narrow { display: none; }

.lbl { width: 5ch; flex: none; color: #94a3b8; }
.c { width: 4ch; flex: none; text-align: center; }
.cols .c { color: #94a3b8; }

.cell {
  font: inherit; color: #64748b; background: transparent; border: 0;
  cursor: pointer; border-radius: 3px; padding: 0;
}
.cell:hover { color: #e2e8f0; background: #334155; }
.cell.on { color: #f8fafc; font-weight: 700; }
.cell:focus-visible { outline: 1px solid #fbbf24; }

.one { color: #7dd3fc; font-weight: 700; }
.zero { color: #94a3b8; }
.probed { background: #713f12; color: #fde68a; border-radius: 3px; }
.comment { color: #64748b; }
.type { color: #f472b6; }
.num { color: #fbbf24; }
.arrow { color: #94a3b8; }
.binary span.sp { margin-left: 1ch; }

@container (max-width: 640px) {
  .term { font-size: 13px; padding: 14px 14px; }
  .block.wide { display: none; }
  .block.narrow { display: block; }
  .lbl { width: 5ch; }
  .c { width: 3.4ch; }
}
</style>
