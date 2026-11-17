<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Five in a Row: Win or Not Yet?</h1>
      <p class="hint">Black has just made five. Whether that wins depends on White's next move.</p>

      <div class="picker" role="tablist" aria-label="Example">
        <button
          v-for="(ex, i) in EXAMPLES"
          :key="i"
          type="button"
          role="tab"
          :aria-selected="pick === i"
          :class="['pick', { active: pick === i }]"
          @click="choose(i)"
        >
          {{ ex.name }}
        </button>
      </div>

      <div class="stage">
        <div class="board">
          <RuleBoard
            :cols="7"
            :rows="5"
            :stones="stones"
            :ghost="current.reply && !replied ? { ...current.reply, v: 2 } : null"
            :ghost-clickable="!!current.reply && !replied"
            :removed="replied ? current.captured : []"
            :lines="lines"
            :fresh="replied ? `${current.reply.c},${current.reply.r}` : ''"
            :label="current.name"
            @play="replied = true"
          />
        </div>

        <div class="side">
          <p :class="['verdict', verdict.kind]">{{ verdict.title }}</p>
          <p class="text" v-html="verdict.text" />
          <div v-if="current.reply" class="actions">
            <button v-if="!replied" type="button" class="act primary" @click="replied = true">Play White's reply ○</button>
            <button v-else type="button" class="act" @click="replied = false">↺ Reset</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

const FIVE = { '1,2': 1, '2,2': 1, '3,2': 1, '4,2': 1, '5,2': 1 };
const GREEN = '#16a34a';
const AMBER = '#d97706';

const EXAMPLES = [
  {
    name: 'Unbreakable five',
    stones: FIVE,
    reply: null,
  },
  {
    name: 'Breakable five',
    stones: { ...FIVE, '1,1': 2, '1,3': 1 },
    reply: { c: 1, r: 4 },
    captured: [{ c: 1, r: 2 }, { c: 1, r: 3 }],
  },
];

const pick = ref(0);
const replied = ref(false);
const current = computed(() => EXAMPLES[pick.value]);

const choose = (i) => {
  pick.value = i;
  replied.value = false;
};

const stones = computed(() => {
  if (!replied.value) return current.value.stones;
  const next = { ...current.value.stones, [`${current.value.reply.c},${current.value.reply.r}`]: 2 };
  for (const { c, r } of current.value.captured) delete next[`${c},${r}`];
  return next;
});

const lines = computed(() => {
  if (pick.value === 0) return [{ from: [1, 2], to: [5, 2], color: GREEN }];
  if (!replied.value) {
    return [
      { from: [1, 2], to: [5, 2], color: AMBER },
      { from: [1, 1], to: [1, 4], color: '#dc2626' },
    ];
  }
  return [{ from: [2, 2], to: [5, 2], color: '#94a3b8' }];
});

const verdict = computed(() => {
  if (pick.value === 0) {
    return {
      kind: 'win',
      title: 'Black wins',
      text: 'No white move can capture a stone of the line, so the five stands and the game ends.',
    };
  }
  if (!replied.value) {
    return {
      kind: 'wait',
      title: 'Not a win yet',
      text: 'The leftmost stone of the five sits in a pair with the stone below it, with White on one side and an empty point on the other. White can capture that pair next move, so the game goes on.',
    };
  }
  return {
    kind: 'broken',
    title: 'The five is broken',
    text: 'White captures the pair. Black is left with <strong>four</strong> in a row and the game continues.',
  };
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
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 12px; }

.picker { display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.pick {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 8px 18px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155; transition: all 0.15s;
}
.pick:hover { border-color: #94a3b8; }
.pick.active { background: #1e293b; border-color: #1e293b; color: #fff; }

.stage { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr); gap: 20px; align-items: center; }
.board { max-width: 420px; width: 100%; justify-self: center; }

.side { display: flex; flex-direction: column; gap: 10px; }
.verdict {
  align-self: flex-start; font-size: 13px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
  padding: 4px 12px; border-radius: 999px;
}
.verdict.win { color: #14532d; background: #dcfce7; }
.verdict.wait { color: #78350f; background: #fef3c7; }
.verdict.broken { color: #7f1d1d; background: #fee2e2; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }
.act {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.act:hover { background: #f1f5f9; border-color: #94a3b8; }
.act.primary { background: #1e293b; border-color: #1e293b; color: #fff; }
.act.primary:hover { background: #334155; }

@container (max-width: 560px) {
  .stage { grid-template-columns: 1fr; gap: 14px; }
}
</style>
