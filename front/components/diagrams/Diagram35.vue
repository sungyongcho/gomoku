<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Capturing a Pair</h1>
      <p class="hint">Click the marked point, or press the button, to play Black's move.</p>

      <div class="stage">
        <div class="board">
          <RuleBoard
            :cols="6"
            :rows="3"
            :stones="played ? AFTER : BEFORE"
            :ghost="played ? null : { c: 4, r: 1, v: 1 }"
            :ghost-clickable="!played"
            :removed="played ? CAPTURED : []"
            :lines="played ? [] : [{ from: [1, 1], to: [4, 1], color: '#2563eb' }]"
            :fresh="played ? '4,1' : ''"
            :label="played ? 'After the capture' : 'Before the capture'"
            @play="played = true"
          />
        </div>

        <div class="side">
          <ol class="steps">
            <li :class="{ on: !played }">1 · Before</li>
            <li :class="{ on: played }">2 · After</li>
          </ol>

          <p class="pattern" aria-label="Pattern: Black, White, White, Black">
            <span class="st b" />
            <span class="st w" :class="{ gone: played }" />
            <span class="st w" :class="{ gone: played }" />
            <span :class="['st', 'b', { pending: !played }]" />
          </p>

          <p v-if="!played" class="text">
            Black to move. A black stone on the marked point closes the line
            <strong>Black, White, White, Black</strong>, so the two white stones in between are captured.
          </p>
          <p v-else class="text">
            The two white stones are removed and Black has captured <strong>one pair</strong>.
            Only the stone just played can capture: a stone that moves in between two enemy stones is safe.
          </p>

          <div class="actions">
            <button v-if="!played" type="button" class="act primary" @click="played = true">Play ● here</button>
            <button v-else type="button" class="act" @click="played = false">↺ Reset</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import RuleBoard from './RuleBoard.vue';

const BEFORE = { '1,1': 1, '2,1': 2, '3,1': 2 };
const AFTER = { '1,1': 1, '4,1': 1 };
const CAPTURED = [{ c: 2, r: 1 }, { c: 3, r: 1 }];

const played = ref(false);
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
.hint { font-size: 13px; color: #64748b; text-align: center; margin: 4px 0 16px; }

.stage { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 20px; align-items: center; }
.board { max-width: 400px; width: 100%; justify-self: center; }

.side { display: flex; flex-direction: column; gap: 12px; }
.steps { list-style: none; display: flex; gap: 6px; }
.steps li {
  font-size: 12px; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 999px; color: #94a3b8; background: #f1f5f9;
}
.steps li.on { color: #1d4ed8; background: #dbeafe; }

.pattern { display: flex; gap: 6px; align-items: center; }
.st { width: 22px; height: 22px; border-radius: 50%; box-shadow: 0 1px 3px 1px #a8a29e; transition: opacity 0.2s; }
.st.b { background: radial-gradient(circle at center, #4b5563, #111827 50%, #000); }
.st.w { background: radial-gradient(circle at center, #fff 60%, #d1d5db); }
.st.pending { opacity: 0.4; box-shadow: 0 0 0 2px #2563eb; }
.st.gone { opacity: 0.15; box-shadow: 0 0 0 1.5px #dc2626; }

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
