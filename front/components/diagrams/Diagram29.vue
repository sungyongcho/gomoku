<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Tree Reuse Across Moves (<code>AdvanceRoot</code> &amp; <code>AlignRoot</code>)</h1>

      <div class="controls-panel">
        <div class="step-controls">
          <button type="button" class="ctrl-btn" :disabled="step === 0" @click="step--">◀ Prev</button>
          <span class="step-badge">Step {{ step + 1 }} / {{ TOTAL_STEPS }}</span>
          <button type="button" class="ctrl-btn" :disabled="step === TOTAL_STEPS - 1" @click="step++">Next ▶</button>
          <button type="button" class="ctrl-btn" @click="resetAll">↺ Reset</button>
        </div>

        <div class="scenario-controls" role="group" aria-label="Incoming request scenario">
          <span class="scenario-label">Opponent / incoming request:</span>
          <button
            v-for="sc in SCENARIOS"
            :key="sc.id"
            type="button"
            class="scen-btn"
            :class="{ active: scenario === sc.id }"
            @click="selectScenario(sc.id)"
          >
            {{ sc.label }}
          </button>
        </div>
      </div>

      <div class="narration" :class="narrationClass">
        <strong>{{ currentStepTitle }}:</strong> {{ currentNarration }}
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 410"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Step-through diagram of tree reuse across two moves showing AdvanceRoot, AlignRoot, and carried-over visits"
      >
        <!-- Illustrative disclaimer -->
        <rect x="16" y="8" width="748" height="26" rx="6" fill="#fffbeb" stroke="#fcd34d" stroke-width="1.2" />
        <text x="390" y="25" text-anchor="middle" font-size="11.5" font-weight="600" fill="#92400e">
          Illustrative toy tree (visit counts 600, 380, 240, 150, 95, 70 are for illustration, not a 19×19 measurement)
        </text>

        <!-- STEP 0 & 1: Top level s0 -> M1, M2, M3 -->
        <g v-if="step <= 1">
          <!-- Edges s0 -> M1, M2, M3 -->
          <line x1="270" y1="98" x2="170" y2="152" :stroke="step === 1 ? '#16a34a' : '#2563eb'" stroke-width="2.5" />
          <line x1="270" y1="98" x2="340" y2="152" stroke="#cbd5e1" stroke-width="1.8" :stroke-dasharray="step === 1 ? '4 3' : undefined" />
          <line x1="270" y1="98" x2="470" y2="152" stroke="#cbd5e1" stroke-width="1.8" :stroke-dasharray="step === 1 ? '4 3' : undefined" />

          <!-- s0 -->
          <rect
            x="185" y="48" width="170" height="50" rx="8"
            :fill="step === 0 ? '#eff6ff' : '#f1f5f9'"
            :stroke="step === 0 ? '#2563eb' : '#94a3b8'"
            :stroke-width="step === 0 ? '2' : '1.5'"
            :stroke-dasharray="step === 1 ? '4 3' : undefined"
          />
          <text x="270" y="69" text-anchor="middle" font-size="13" font-weight="700" :fill="step === 0 ? '#1e3a8a' : '#64748b'">
            {{ step === 0 ? 'Root s₀ (our turn)' : 'Old root s₀ (freed)' }}
          </text>
          <text x="270" y="87" text-anchor="middle" font-size="12" :fill="step === 0 ? '#1e40af' : '#64748b'">
            N = 600 new searches
          </text>

          <!-- M1 (chosen) -->
          <rect
            x="95" y="152" width="150" height="54" rx="8"
            :fill="step === 1 ? '#dcfce7' : '#eff6ff'"
            :stroke="step === 1 ? '#16a34a' : '#2563eb'"
            stroke-width="2.2"
          />
          <text x="170" y="173" text-anchor="middle" font-size="13" font-weight="700" :fill="step === 1 ? '#166534' : '#1e3a8a'">
            Move M1 {{ step === 1 ? '→ New Root' : '(picked)' }}
          </text>
          <text x="170" y="192" text-anchor="middle" font-size="12" font-weight="600" :fill="step === 1 ? '#15803d' : '#1e40af'">
            N = 380 · kept
          </text>

          <!-- M2 (sibling) -->
          <rect
            x="275" y="152" width="130" height="54" rx="8"
            :fill="step === 1 ? '#fef2f2' : '#ffffff'"
            :stroke="step === 1 ? '#fca5a5' : '#cbd5e1'"
            stroke-width="1.5"
            :stroke-dasharray="step === 1 ? '4 3' : undefined"
          />
          <text x="340" y="173" text-anchor="middle" font-size="12.5" font-weight="600" :fill="step === 1 ? '#991b1b' : '#334155'">
            Move M2
          </text>
          <text x="340" y="192" text-anchor="middle" font-size="11.5" :fill="step === 1 ? '#b91c1c' : '#64748b'">
            N = 150 {{ step === 1 ? '· freed' : '' }}
          </text>

          <!-- M3 (sibling) -->
          <rect
            x="418" y="152" width="124" height="54" rx="8"
            :fill="step === 1 ? '#fef2f2' : '#ffffff'"
            :stroke="step === 1 ? '#fca5a5' : '#cbd5e1'"
            stroke-width="1.5"
            :stroke-dasharray="step === 1 ? '4 3' : undefined"
          />
          <text x="480" y="173" text-anchor="middle" font-size="12.5" font-weight="600" :fill="step === 1 ? '#991b1b' : '#334155'">
            Move M3
          </text>
          <text x="480" y="192" text-anchor="middle" font-size="11.5" :fill="step === 1 ? '#b91c1c' : '#64748b'">
            N = 70 {{ step === 1 ? '· freed' : '' }}
          </text>

          <!-- Edges M1 -> R1, R2, R3 -->
          <line x1="170" y1="206" x2="90" y2="264" stroke="#64748b" stroke-width="1.8" />
          <line x1="170" y1="206" x2="245" y2="264" stroke="#64748b" stroke-width="1.8" />
          <line x1="170" y1="206" x2="400" y2="264" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3" />

          <!-- R1 -->
          <rect x="20" y="264" width="140" height="58" rx="8" fill="#ffffff" stroke="#64748b" stroke-width="1.6" />
          <text x="90" y="285" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">Reply R1</text>
          <text x="90" y="302" text-anchor="middle" font-size="11.5" fill="#1e40af">N = 240 (expected)</text>
          <text x="90" y="316" text-anchor="middle" font-size="10.5" fill="#64748b">subtree kept</text>

          <!-- R2 -->
          <rect x="175" y="264" width="140" height="58" rx="8" fill="#ffffff" stroke="#64748b" stroke-width="1.6" />
          <text x="245" y="285" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">Reply R2</text>
          <text x="245" y="302" text-anchor="middle" font-size="11.5" fill="#334155">N = 95</text>
          <text x="245" y="316" text-anchor="middle" font-size="10.5" fill="#64748b">subtree kept</text>

          <!-- R3 -->
          <rect x="330" y="264" width="145" height="58" rx="8" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="4 3" />
          <text x="402" y="285" text-anchor="middle" font-size="12.5" font-weight="700" fill="#475569">Reply R3</text>
          <text x="402" y="302" text-anchor="middle" font-size="11.5" fill="#64748b">N = 0 (unvisited)</text>
          <text x="402" y="316" text-anchor="middle" font-size="10.5" fill="#64748b">MctsNode* is nullptr</text>

          <!-- Side explanation card -->
          <rect x="556" y="48" width="208" height="274" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
          <text x="660" y="74" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
            {{ step === 0 ? '1. Answer Search' : '2. AdvanceRoot(M1)' }}
          </text>
          <line x1="572" y1="84" x2="748" y2="84" stroke="#e2e8f0" stroke-width="1" />
          <template v-if="step === 0">
            <text x="572" y="108" font-size="11.5" fill="#334155">• Root s₀ starts with 600</text>
            <text x="572" y="124" font-size="11.5" fill="#334155">  illustrative searches.</text>
            <text x="572" y="148" font-size="11.5" fill="#334155">• Engine picks M1 (380 visits,</text>
            <text x="572" y="164" font-size="11.5" fill="#334155">  most searched).</text>
            <text x="572" y="188" font-size="11.5" fill="#334155">• Only visited moves have</text>
            <text x="572" y="204" font-size="11.5" fill="#334155">  MctsNode allocations;</text>
            <text x="572" y="220" font-size="11.5" fill="#334155">  unvisited R3 stays null.</text>
          </template>
          <template v-else>
            <text x="572" y="108" font-size="11.5" fill="#166534" font-weight="600">• Answer sent to client.</text>
            <text x="572" y="132" font-size="11.5" fill="#334155">• M1 becomes root_ (N=380).</text>
            <text x="572" y="156" font-size="11.5" fill="#991b1b">• Old root s₀ and siblings</text>
            <text x="572" y="172" font-size="11.5" fill="#991b1b">  M2, M3 are deleted.</text>
            <text x="572" y="196" font-size="11.5" fill="#334155">• node_count_ is recounted</text>
            <text x="572" y="212" font-size="11.5" fill="#334155">  from the kept subtree.</text>
            <text x="572" y="236" font-size="11.5" fill="#1e40af">• Replies R1, R2, R3 sit</text>
            <text x="572" y="252" font-size="11.5" fill="#1e40af">  one ply below root_.</text>
          </template>
        </g>

        <!-- STEP 2 & 3: AlignRoot + Next Search -->
        <g v-else>
          <!-- Kept root M1 before alignment -->
          <rect
            x="160" y="48" width="210" height="52" rx="8"
            :fill="scenario === 'unexpected' ? '#fef2f2' : '#f1f5f9'"
            :stroke="scenario === 'unexpected' ? '#ef4444' : '#94a3b8'"
            stroke-width="1.8"
            stroke-dasharray="4 3"
          />
          <text x="265" y="70" text-anchor="middle" font-size="12.5" font-weight="700" :fill="scenario === 'unexpected' ? '#991b1b' : '#475569'">
            Kept root M1 (N = 380)
          </text>
          <text x="265" y="88" text-anchor="middle" font-size="11.5" :fill="scenario === 'unexpected' ? '#b91c1c' : '#64748b'">
            {{ scenario === 'unexpected' ? 'dropped by reset_root()' : 'freed after aligning to reply' }}
          </text>

          <!-- Edges from M1 -->
          <line x1="210" y1="100" x2="95" y2="158" :stroke="scenario === 'expected' ? '#16a34a' : '#cbd5e1'" :stroke-width="scenario === 'expected' ? '2.5' : '1.5'" :stroke-dasharray="scenario !== 'expected' ? '4 3' : undefined" />
          <line x1="265" y1="100" x2="245" y2="158" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="4 3" />
          <line x1="320" y1="100" x2="405" y2="158" :stroke="scenario === 'unsearched' ? '#2563eb' : '#cbd5e1'" :stroke-width="scenario === 'unsearched' ? '2.5' : '1.5'" stroke-dasharray="4 3" />

          <!-- R1 node -->
          <rect
            x="20" y="158" width="150" height="72" rx="8"
            :fill="scenario === 'expected' ? '#dcfce7' : '#fef2f2'"
            :stroke="scenario === 'expected' ? '#16a34a' : '#fca5a5'"
            :stroke-width="scenario === 'expected' ? '2.4' : '1.5'"
            :stroke-dasharray="scenario === 'expected' ? undefined : '4 3'"
          />
          <text x="95" y="180" text-anchor="middle" font-size="13" font-weight="700" :fill="scenario === 'expected' ? '#166534' : '#991b1b'">
            Reply R1 {{ scenario === 'expected' ? '✓ Match' : '' }}
          </text>
          <text x="95" y="198" text-anchor="middle" font-size="11.5" :fill="scenario === 'expected' ? '#15803d' : '#b91c1c'">
            {{ scenario === 'expected' ? 'carried: 240' : 'N = 240 · freed' }}
          </text>
          <text v-if="scenario === 'expected' && step === 3" x="95" y="216" text-anchor="middle" font-size="11.5" font-weight="700" fill="#166534">
            + 600 new = 840
          </text>
          <text v-else-if="scenario === 'expected'" x="95" y="216" text-anchor="middle" font-size="11" fill="#166534">
            new root_
          </text>

          <!-- R2 node -->
          <rect x="182" y="158" width="126" height="72" rx="8" fill="#fef2f2" stroke="#fca5a5" stroke-width="1.5" stroke-dasharray="4 3" />
          <text x="245" y="180" text-anchor="middle" font-size="12.5" font-weight="600" fill="#991b1b">Reply R2</text>
          <text x="245" y="198" text-anchor="middle" font-size="11.5" fill="#b91c1c">N = 95 · freed</text>

          <!-- R3 node -->
          <rect
            x="320" y="158" width="170" height="72" rx="8"
            :fill="scenario === 'unsearched' ? '#eff6ff' : '#fef2f2'"
            :stroke="scenario === 'unsearched' ? '#2563eb' : '#fca5a5'"
            :stroke-width="scenario === 'unsearched' ? '2.4' : '1.5'"
            :stroke-dasharray="scenario === 'unsearched' ? undefined : '4 3'"
          />
          <text x="405" y="180" text-anchor="middle" font-size="13" font-weight="700" :fill="scenario === 'unsearched' ? '#1e3a8a' : '#991b1b'">
            Reply R3 {{ scenario === 'unsearched' ? '✓ Match' : '' }}
          </text>
          <text x="405" y="198" text-anchor="middle" font-size="11.5" :fill="scenario === 'unsearched' ? '#1e40af' : '#b91c1c'">
            {{ scenario === 'unsearched' ? 'carried: 0 (allocated now)' : 'N = 0 · discarded' }}
          </text>
          <text v-if="scenario === 'unsearched' && step === 3" x="405" y="216" text-anchor="middle" font-size="11.5" font-weight="700" fill="#1e3a8a">
            + 600 new = 600
          </text>
          <text v-else-if="scenario === 'unsearched'" x="405" y="216" text-anchor="middle" font-size="11" fill="#1e40af">
            node_count = 1
          </text>

          <!-- Unexpected fresh root box (when scenario === 'unexpected') -->
          <g v-if="scenario === 'unexpected'">
            <rect x="85" y="256" width="340" height="74" rx="8" fill="#fffbeb" stroke="#d97706" stroke-width="2.2" />
            <text x="255" y="279" text-anchor="middle" font-size="13" font-weight="700" fill="#92400e">
              AlignRoot returned -1 → reset_root(target)
            </text>
            <text x="255" y="298" text-anchor="middle" font-size="12" fill="#b45309">
              carried: 0 · fresh single-node tree started
            </text>
            <text v-if="step === 3" x="255" y="317" text-anchor="middle" font-size="12" font-weight="700" fill="#92400e">
              0 + 600 new = 600 total searches
            </text>
          </g>

          <!-- Subtree continuation bar when expected or unsearched -->
          <g v-else>
            <rect x="20" y="256" width="470" height="74" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
            <text x="255" y="279" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
              {{ step === 2 ? '3. AlignRoot Result' : '4. ContinueSimulationEncoded (+600 new)' }}
            </text>
            <text x="255" y="300" text-anchor="middle" font-size="12" fill="#334155">
              <template v-if="scenario === 'expected'">
                {{ step === 2 ? 'Carried over: 240 searches already rooted at R1' : 'N_move = N_carried + N_new = 240 + 600 = 840' }}
              </template>
              <template v-else>
                {{ step === 2 ? 'Carried over: 0 searches (R3 had no prior visits, but matched)' : 'N_move = N_carried + N_new = 0 + 600 = 600' }}
              </template>
            </text>
            <text x="255" y="318" text-anchor="middle" font-size="11" fill="#64748b">
              Siblings below M1 are deleted; only the matched subtree stays in memory.
            </text>
          </g>

          <!-- SameState 7-field checklist card -->
          <rect x="514" y="48" width="250" height="282" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
          <text x="639" y="72" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">
            SameState() Check (7 fields)
          </text>
          <line x1="530" y1="82" x2="748" y2="82" stroke="#e2e8f0" stroke-width="1" />
          <g v-for="(field, idx) in STATE_FIELDS" :key="field.name">
            <text x="532" :y="106 + idx * 27" font-size="11.5" font-family="monospace" fill="#334155">
              {{ field.name }}
            </text>
            <text
              x="746"
              :y="106 + idx * 27"
              text-anchor="end"
              font-size="11.5"
              font-weight="700"
              :fill="fieldStatusColor(field.mismatch)"
            >
              {{ fieldStatusText(field.mismatch) }}
            </text>
          </g>
          <line x1="530" y1="292" x2="748" y2="292" stroke="#e2e8f0" stroke-width="1" />
          <text
            x="639"
            y="314"
            text-anchor="middle"
            font-size="12"
            font-weight="700"
            :fill="scenario === 'unexpected' ? '#dc2626' : '#15803d'"
          >
            {{ scenario === 'unexpected' ? 'Returns -1 (start new tree)' : 'All 7 match → reuse subtree' }}
          </text>
        </g>

        <!-- Bottom summary formula bar -->
        <rect x="16" y="348" width="748" height="48" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="390" y="369" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
          N_move = N_carried + N_new = {{ summaryCarried }} + {{ summaryNew }} = {{ summaryCarried + summaryNew }}
        </text>
        <text x="390" y="387" text-anchor="middle" font-size="11.5" fill="#475569">
          {{ summarySubtext }}
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const TOTAL_STEPS = 4;

const SCENARIOS = [
  { id: 'expected', label: 'Expected reply (R1)' },
  { id: 'unsearched', label: 'Unsearched reply (R3)' },
  { id: 'unexpected', label: 'Mismatch (takeback / edit)' },
];

const STATE_FIELDS = [
  { name: 'next_player', mismatch: false },
  { name: 'last_move_idx', mismatch: false },
  { name: 'p1_pts', mismatch: false },
  { name: 'p2_pts', mismatch: false },
  { name: 'empty_count', mismatch: true },
  { name: 'board[361]', mismatch: true },
  { name: 'history[<=5]', mismatch: true },
];

const step = ref(0);
const scenario = ref('expected');

function selectScenario(id) {
  scenario.value = id;
  if (step.value < 2) {
    step.value = 2;
  }
}

function resetAll() {
  step.value = 0;
  scenario.value = 'expected';
}

const currentStepTitle = computed(() => {
  if (step.value === 0) return 'Step 1 — Answer search';
  if (step.value === 1) return 'Step 2 — Commit move (AdvanceRoot)';
  if (step.value === 2) return 'Step 3 — Incoming request (AlignRoot)';
  return 'Step 4 — Continue search';
});

const currentNarration = computed(() => {
  if (step.value === 0) {
    return 'The engine runs an answer search from s₀ (600 illustrative searches) and picks the most-visited move M1 (380 visits). Unvisited moves such as R3 have no MctsNode allocated yet.';
  }
  if (step.value === 1) {
    return 'After sending M1, SearchSession.commit_move calls AdvanceRoot(M1): M1 becomes the new root (N=380), and s₀ plus sibling branches M2 and M3 are deleted.';
  }
  if (step.value === 2) {
    if (scenario.value === 'expected') {
      return 'Opponent plays R1. AlignRoot applies R1 to M1, confirms all 7 GomokuState fields match, and promotes R1 to root with 240 carried-over searches.';
    }
    if (scenario.value === 'unsearched') {
      return 'Opponent plays R3. All 7 fields match M1 + R3, so AlignRoot succeeds even though R3 had 0 visits (nullptr child) and creates a 1-node root (0 carried over).';
    }
    return 'Request does not match M1 or any move one ply below it (e.g., takeback, board edit, capture score or history mismatch). AlignRoot returns -1 and reset_root starts fresh.';
  }
  if (scenario.value === 'expected') {
    return 'ContinueSimulationEncoded adds 600 new searches onto the 240 carried-over searches, backing the answer with 840 total searches.';
  }
  if (scenario.value === 'unsearched') {
    return 'ContinueSimulationEncoded adds 600 new searches to the newly allocated R3 root (0 carried + 600 new = 600 total).';
  }
  return 'ContinueSimulationEncoded runs 600 new searches on the fresh tree (0 carried + 600 new = 600 total).';
});

const narrationClass = computed(() => {
  if (step.value === 1) return 'hl-green';
  if (step.value >= 2) {
    if (scenario.value === 'expected') return 'hl-green';
    if (scenario.value === 'unsearched') return 'hl-blue';
    return 'hl-amber';
  }
  return 'hl-blue';
});

const summaryCarried = computed(() => {
  if (step.value < 2) return 0;
  return scenario.value === 'expected' ? 240 : 0;
});

const summaryNew = computed(() => {
  if (step.value === 0 || step.value === 3) return 600;
  return 0;
});

const summarySubtext = computed(() => {
  if (step.value === 0) return 'First move on the connection: fresh root s₀.';
  if (step.value === 1) return 'Between moves: kept root is M1 (380 visits), waiting for the next request.';
  if (step.value === 2) return 'Ready for the timed search on the aligned root.';
  if (scenario.value === 'expected') return 'Carried-over subtree saves work from the previous turn (illustrative values).';
  return '0 carried over; timed search starts from an empty root.';
});

function fieldStatusColor(isMismatchInUnexpected) {
  if (scenario.value !== 'unexpected') return '#15803d';
  return isMismatchInUnexpected ? '#dc2626' : '#64748b';
}

function fieldStatusText(isMismatchInUnexpected) {
  if (scenario.value !== 'unexpected') return '✓ match';
  return isMismatchInUnexpected ? '✗ differs' : 'checked';
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

.controls-panel {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 10px;
}

.step-controls,
.scenario-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.scenario-label {
  font-size: 12.5px;
  font-weight: 600;
  color: #475569;
}

.ctrl-btn,
.scen-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 5px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.ctrl-btn:hover:not(:disabled),
.scen-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.scen-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.step-badge {
  font-size: 12.5px;
  font-weight: 600;
  color: #475569;
  min-width: 82px;
  text-align: center;
}

.narration {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  line-height: 1.45;
  color: #334155;
  margin-bottom: 10px;
}

.narration.hl-blue {
  background: #eff6ff;
  border-color: #93c5fd;
}

.narration.hl-green {
  background: #f0fdf4;
  border-color: #86efac;
}

.narration.hl-amber {
  background: #fffbeb;
  border-color: #fcd34d;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
