<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Settings Dialog</h1>
      <p class="hint">The game's own controls, wired to a local copy of the settings. Try switching to <code>alphazero</code> or turning captures off.</p>

      <section class="dialog">
        <header class="dialog-head">
          <span class="dialog-title">Settings</span>
          <button type="button" class="reset" :disabled="isDefault" @click="reset">↺ Reset to defaults</button>
        </header>

        <form class="grid" @submit.prevent>
          <div :class="{ wide: isAlphaZero }">
            <BButtonGroupSwitch
              v-model="s.ai"
              :options="[{ value: 'minimax', label: 'minimax' }, { value: 'alphazero', label: 'alphazero' }]"
              label="AI model"
            />
            <div v-if="isAlphaZero" class="az-note">
              <p>AlphaZero mode: MCTS searches for 2.5 seconds per move. During training, each move used 400 to 800 searches.</p>
              <p>Also, evaluation is not supported in AlphaZero mode.</p>
            </div>
          </div>

          <BButtonGroupSwitch
            v-if="!isAlphaZero"
            v-model="s.difficulty"
            :options="[{ value: 'easy', label: 'easy' }, { value: 'medium', label: 'medium' }, { value: 'hard', label: 'hard' }]"
            label="Difficulty"
          />

          <BButtonGroupSwitch
            v-model="s.firstMove"
            :options="[{ value: 'Player1', label: 'Player1' }, { value: 'Player2', label: 'Player2' }]"
            label="First move"
          />

          <BToggleSwitch v-model="s.enableDoubleThreeRestriction" label="Enable Double-Three Prohibition" :disabled="isAlphaZero" />
          <BToggleSwitch v-model="s.enableCapture" label="Enable Capture" :disabled="isAlphaZero" />

          <BButtonGroupSwitch
            v-if="s.enableCapture"
            v-model="s.totalPairCaptured"
            :options="[3, 4, 5, 6, 7].map((n) => ({ value: n, label: String(n) }))"
            label="Total of captured pair stones"
            :disabled="isAlphaZero"
          />
          <BSlider v-if="s.enableCapture" v-model="s.advantage1" :max="s.totalPairCaptured - 1" label="Player1 Advantage" :disabled="isAlphaZero" />
          <BSlider v-if="s.enableCapture" v-model="s.advantage2" :max="s.totalPairCaptured - 1" label="Player2 Advantage" :disabled="isAlphaZero" />
        </form>
      </section>

      <p class="log" aria-live="polite">{{ log }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed, reactive, ref, watch } from 'vue';

// Same values as defaultSettings in stores/game.store.ts.
const DEFAULTS = {
  ai: 'minimax',
  difficulty: 'hard',
  firstMove: 'Player1',
  enableDoubleThreeRestriction: true,
  enableCapture: true,
  totalPairCaptured: 5,
  advantage1: 0,
  advantage2: 0,
};

const s = reactive({ ...DEFAULTS });
const log = ref('Defaults loaded.');
const isAlphaZero = computed(() => s.ai === 'alphazero');
const isDefault = computed(() => Object.keys(DEFAULTS).every((k) => s[k] === DEFAULTS[k]));

// The same two rules as SettingModal.vue.
watch(() => s.totalPairCaptured, () => {
  if (s.advantage1 || s.advantage2) log.value = 'Capture goal changed, so both advantages went back to 0.';
  s.advantage1 = 0;
  s.advantage2 = 0;
});
watch(isAlphaZero, (v) => {
  if (!v) {
    log.value = 'minimax: every setting can be changed again.';
    return;
  }
  Object.assign(s, { enableDoubleThreeRestriction: true, enableCapture: true, totalPairCaptured: 5, advantage1: 0, advantage2: 0 });
  log.value = 'alphazero: Difficulty is hidden and the rules are locked to the defaults. Only First move can change.';
});
watch(() => s.enableCapture, (v) => {
  if (!isAlphaZero.value) log.value = v ? 'Captures on: the capture goal and advantages are back.' : 'Captures off: the capture goal and advantages are hidden.';
});

const reset = () => {
  Object.assign(s, DEFAULTS);
  log.value = 'Defaults loaded.';
};
</script>

<style scoped>
.diagram-wrapper {
  font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  padding: 24px 16px;
  container-type: inline-size;
}
.container { width: 100%; max-width: 820px; margin: 0 auto; }
h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; margin: 0; }
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 16px; }
.hint code { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.92em; }

.dialog { background: #fff; border-radius: 12px; box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.25), 0 0 0 1px #e2e8f0; padding: 20px 22px 26px; }
.dialog-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 20px; }
.dialog-title { font-size: 20px; font-weight: 700; color: #334155; }
.reset {
  font-family: inherit; font-size: 12.5px; font-weight: 600;
  padding: 6px 12px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.reset:hover:not(:disabled) { background: #f1f5f9; border-color: #94a3b8; }
.reset:disabled { opacity: 0.4; cursor: default; }

.grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px 32px; font-size: 15px; color: #334155; }
.wide { grid-column: 1 / -1; }
.az-note { margin-top: 14px; padding: 12px 14px; border-radius: 8px; border: 1px solid #bfdbfe; background: #eff6ff; color: #1e3a8a; font-size: 13.5px; font-weight: 500; line-height: 1.5; }
.az-note p + p { margin-top: 6px; }

.log { margin: 12px 0 0; font-size: 13px; color: #475569; text-align: center; min-height: 1.5em; }

@container (max-width: 560px) {
  .dialog { padding: 16px 14px 20px; }
  .grid { grid-template-columns: 1fr; gap: 22px; }
}
</style>
