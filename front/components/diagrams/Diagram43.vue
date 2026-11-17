<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Winning by Capture</h1>
      <p class="hint">Each button plays one capture. The first player to reach the capture goal wins at once.</p>

      <div class="stage">
        <aside class="match">
          <div class="match-head">
            <span class="caps">Match</span>
            <span class="settings"><i class="pi pi-cog" /> Settings</span>
          </div>
          <div class="players">
            <div class="player">
              <span :class="['frame', { active: turn === 'X' && !winner }]">
                <img :src="player1Image" alt="" class="avatar black" />
              </span>
              <span class="name">Player 1</span>
            </div>
            <div class="vs">
              <span class="vs-word">vs</span>
              <span class="turn">Turn {{ Math.floor(moves / 2) }}</span>
            </div>
            <div class="player">
              <span :class="['frame', { active: turn === 'O' && !winner }]">
                <img :src="aiImage" alt="" class="avatar white" />
              </span>
              <span class="name">AI</span>
            </div>
          </div>
          <div :class="['score', { done: winner }]">
            <span class="num"><b :class="{ hit: winner === 'X' }">{{ p1 }}</b><small>/{{ goal }}</small></span>
            <span class="caps">Pair captured</span>
            <span class="num"><b :class="{ hit: winner === 'O' }">{{ p2 }}</b><small>/{{ goal }}</small></span>
          </div>
        </aside>

        <div class="side">
          <div v-if="winner" class="alert" role="alert">
            <p class="alert-head"><i class="pi pi-info-circle" /> Game Over</p>
            <p class="alert-msg">{{ winner === 'X' ? 'Black' : 'White' }} Win - Captured {{ goal }}</p>
          </div>
          <template v-else>
            <div class="goal">
              <span class="goal-label">Capture goal</span>
              <div class="seg">
                <button
                  v-for="g in [3, 4, 5, 6, 7]"
                  :key="g"
                  type="button"
                  :class="{ on: goal === g }"
                  :disabled="g <= Math.max(p1, p2)"
                  @click="goal = g"
                >{{ g }}</button>
              </div>
            </div>
            <div class="actions">
              <button type="button" class="act solid" @click="capture('X')">● Player 1 captures a pair</button>
              <button type="button" class="act" @click="capture('O')">○ AI captures a pair</button>
            </div>
          </template>
          <p class="text">
            Player 1 plays Black here. A capture removes two stones and adds one pair to the capturer's count.
            The default goal is <strong>5 pairs</strong>; Settings allows 3 to 7.
          </p>
          <button type="button" class="act" :disabled="isStart" @click="reset">↺ Reset</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';
import player1Image from '~/assets/player1.webp';
import aiImage from '~/assets/ai.webp';

const START = { p1: 3, p2: 1, moves: 44, goal: 5 };
const p1 = ref(START.p1);
const p2 = ref(START.p2);
const moves = ref(START.moves);
const goal = ref(START.goal);
const turn = ref('X');

const winner = computed(() => (p1.value >= goal.value ? 'X' : p2.value >= goal.value ? 'O' : ''));
const isStart = computed(() => p1.value === START.p1 && p2.value === START.p2 && goal.value === START.goal);

const capture = (who) => {
  if (who === 'X') p1.value++;
  else p2.value++;
  moves.value += 2; // the capturing move and the reply before it
  turn.value = who === 'X' ? 'O' : 'X';
};
const reset = () => {
  p1.value = START.p1;
  p2.value = START.p2;
  moves.value = START.moves;
  goal.value = START.goal;
  turn.value = 'X';
};
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
.container { width: 100%; max-width: 760px; margin: 0 auto; }
h1 { font-size: 19px; font-weight: 700; text-align: center; color: #0f172a; }
.hint { font-size: 13px; line-height: 1.5; color: #64748b; text-align: center; margin: 4px 0 16px; }

.stage { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr); gap: 24px; align-items: center; justify-content: center; }

/* Copies InfoBoard's match section (zinc panel, avatars, CapturedScore). */
.match { background: #f4f4f5; border-radius: 12px; padding: 16px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); display: flex; flex-direction: column; gap: 12px; }
.match-head { display: flex; justify-content: space-between; align-items: center; }
.caps { font-size: 12px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; color: #71717a; }
.settings { font-size: 12px; color: #52525b; display: inline-flex; gap: 6px; align-items: center; }
.players { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.player { display: flex; flex-direction: column; align-items: center; gap: 6px; }
.frame { border: 6px solid transparent; border-radius: 50%; transition: border-color 0.2s; display: flex; }
.frame.active { border-color: #eab308; }
.avatar { width: 64px; height: 64px; border-radius: 50%; border: 4px solid; display: block; object-fit: cover; }
.avatar.black { border-color: #000; box-shadow: 0 0 2px 1px #fff; }
.avatar.white { border-color: #fff; box-shadow: 0 0 2px 1px #000; }
.name { font-size: 14px; font-weight: 500; color: #3f3f46; }
.vs { display: flex; flex-direction: column; align-items: center; }
.vs-word { font-size: 18px; font-weight: 500; color: #a1a1aa; }
.turn { font-size: 12px; color: #71717a; }
.score { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 6px 10px; border-radius: 6px; border: 1px solid #e4e4e7; background: rgba(250, 250, 250, 0.8); transition: border-color 0.2s; }
.score.done { border-color: #eab308; }
.num b { font-size: 24px; font-weight: 500; color: #27272a; transition: color 0.2s; }
.num b.hit { color: #ca8a04; }
.num small { font-size: 14px; color: #71717a; margin-left: 2px; }

.side { display: flex; flex-direction: column; gap: 12px; align-items: flex-start; }
.goal { display: flex; flex-direction: column; gap: 6px; }
.goal-label { font-size: 13px; font-weight: 600; color: #475569; }
.seg { display: inline-flex; border-radius: 6px; overflow: hidden; border: 1px solid #e2e8f0; }
.seg button { font-family: inherit; font-size: 14px; padding: 6px 12px; border: 0; background: #f1f5f9; color: #334155; cursor: pointer; }
.seg button.on { background: #000; color: #fff; }
.seg button:disabled:not(.on) { opacity: 0.35; cursor: default; }
.actions { display: flex; flex-wrap: wrap; gap: 8px; }
.act {
  font-family: inherit; font-size: 13px; font-weight: 600;
  padding: 7px 14px; border-radius: 999px; cursor: pointer;
  border: 1.5px solid #cbd5e1; background: #fff; color: #334155;
}
.act:hover:not(:disabled) { background: #f1f5f9; border-color: #94a3b8; }
.act:disabled { opacity: 0.4; cursor: default; }
.act.solid { background: #18181b; border-color: #18181b; color: #fff; }
.act.solid:hover { background: #3f3f46; }
.alert { width: 100%; border-radius: 8px; border: 1px solid #bfdbfe; background: #eff6ff; padding: 10px 14px; }
.alert-head { font-size: 14px; font-weight: 700; color: #1e3a8a; display: flex; gap: 6px; align-items: center; }
.alert-msg { font-size: 14px; color: #1e40af; margin-top: 2px; }
.text { font-size: 14px; line-height: 1.6; color: #334155; }

@container (max-width: 600px) {
  .stage { grid-template-columns: 1fr; justify-items: center; }
  .match { width: 100%; max-width: 320px; }
  .side { width: 100%; }
}
</style>
