<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Oracle A1 Deployment &amp; Verification Pipeline</h1>

      <div class="controls">
        <button
          type="button"
          class="ctrl-btn"
          :disabled="currentStep <= 0"
          @click="currentStep -= 1"
        >
          ◀ Prev
        </button>
        <span class="step-label">Step {{ currentStep + 1 }} / {{ STEPS.length }}</span>
        <button
          type="button"
          class="ctrl-btn"
          :disabled="currentStep >= STEPS.length - 1"
          @click="currentStep += 1"
        >
          Next ▶
        </button>
        <button type="button" class="ctrl-btn" @click="currentStep = 0">
          ↺ Reset
        </button>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 760 220"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Five-step Oracle A1 deployment and verification pipeline"
      >
        <defs>
          <marker id="d27-arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#64748b" />
          </marker>
          <marker id="d27-arr-active" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#2563eb" />
          </marker>
        </defs>

        <!-- Zone banners -->
        <rect x="14" y="12" width="440" height="32" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1" />
        <text x="234" y="32" text-anchor="middle" font-size="12" font-weight="700" fill="#475569">
          Workstation → SSH (tcp/22) → Oracle A1 Instance
        </text>

        <rect x="462" y="12" width="138" height="32" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
        <text x="531" y="32" text-anchor="middle" font-size="12" font-weight="700" fill="#065f46">
          SSH Tunnel Check
        </text>

        <rect x="608" y="12" width="138" height="32" rx="6" fill="#fff7ed" stroke="#fed7aa" stroke-width="1" />
        <text x="677" y="32" text-anchor="middle" font-size="12" font-weight="700" fill="#9a3412">
          Cloudflare Edge
        </text>

        <!-- Connecting arrows between steps -->
        <line
          v-for="idx in 4"
          :key="'edge-' + idx"
          :x1="stepX(idx - 1) + BOX_W + 2"
          y1="120"
          :x2="stepX(idx) - 4"
          y2="120"
          :stroke="currentStep === idx ? '#2563eb' : '#94a3b8'"
          :stroke-width="currentStep === idx ? '2.5' : '1.8'"
          :marker-end="currentStep === idx ? 'url(#d27-arr-active)' : 'url(#d27-arr)'"
        />

        <!-- Step boxes -->
        <g
          v-for="(step, idx) in STEPS"
          :key="step.id"
          class="step-node"
          @click="currentStep = idx"
        >
          <rect
            :x="stepX(idx)"
            y="60"
            :width="BOX_W"
            height="124"
            rx="8"
            :fill="nodeFill(idx)"
            :stroke="nodeStroke(idx)"
            :stroke-width="currentStep === idx ? '2.5' : '1.5'"
          />

          <!-- Step number pill -->
          <circle
            :cx="stepX(idx) + 20"
            cy="80"
            r="11"
            :fill="currentStep === idx ? '#2563eb' : idx < currentStep ? '#16a34a' : '#64748b'"
          />
          <text
            :x="stepX(idx) + 20"
            y="84"
            text-anchor="middle"
            font-size="11.5"
            font-weight="700"
            fill="#ffffff"
          >
            {{ idx + 1 }}
          </text>

          <text
            :x="stepX(idx) + 37"
            y="84"
            font-size="12.5"
            font-weight="700"
            :fill="currentStep === idx ? '#1e3a8a' : '#0f172a'"
          >
            {{ step.shortTitle }}
          </text>

          <text
            :x="stepX(idx) + BOX_W / 2"
            y="107"
            text-anchor="middle"
            font-size="11"
            font-weight="600"
            fill="#334155"
          >
            {{ step.scriptLabel }}
          </text>

          <text
            :x="stepX(idx) + BOX_W / 2"
            y="128"
            text-anchor="middle"
            font-size="11"
            fill="#475569"
          >
            {{ step.summaryLine1 }}
          </text>
          <text
            :x="stepX(idx) + BOX_W / 2"
            y="143"
            text-anchor="middle"
            font-size="11"
            fill="#475569"
          >
            {{ step.summaryLine2 }}
          </text>

          <!-- Check badge inside card -->
          <rect
            :x="stepX(idx) + 10"
            y="154"
            :width="BOX_W - 20"
            height="21"
            rx="4"
            :fill="currentStep === idx ? '#dbeafe' : '#f8fafc'"
            :stroke="currentStep === idx ? '#93c5fd' : '#cbd5e1'"
            stroke-width="1"
          />
          <text
            :x="stepX(idx) + BOX_W / 2"
            y="168.5"
            text-anchor="middle"
            font-size="10.5"
            font-weight="600"
            :fill="currentStep === idx ? '#1e40af' : '#475569'"
          >
            {{ step.checkBadge }}
          </text>
        </g>

        <!-- Bottom note bar -->
        <text x="380" y="206" text-anchor="middle" font-size="11.5" fill="#64748b">
          Click any step box above or use Prev / Next to inspect actions, checks, and scripts.
        </text>
      </svg>

      <div class="detail-grid">
        <div class="detail-card">
          <div class="detail-heading">What {{ activeStep.title }} does</div>
          <p class="detail-text">{{ activeStep.does }}</p>
        </div>
        <div class="detail-card check-card">
          <div class="detail-heading">What it checks</div>
          <p class="detail-text">{{ activeStep.checks }}</p>
        </div>
      </div>

      <div class="meta-bar">
        <span><strong>Script / command:</strong> <code>{{ activeStep.command }}</code></span>
        <span><strong>Key settings:</strong> {{ activeStep.settings }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const BOX_W = 136;
const GAP = 13;
const START_X = 14;

function stepX(idx) {
  return START_X + idx * (BOX_W + GAP);
}

const STEPS = [
  {
    id: 'host-setup',
    shortTitle: 'Host Setup',
    title: '1. Host Setup',
    scriptLabel: '01_host_setup.sh',
    summaryLine1: 'Docker, compose,',
    summaryLine2: '2 GB swap, iptables',
    checkBadge: 'SSH + uname/free',
    command: 'bash deploy/alphazero-oracle/01_host_setup.sh',
    settings: 'Canonical Ubuntu 24.04 aarch64, port 8080, ~/build',
    does: 'Uploads and runs host_setup.sh as root: installs Docker, docker-compose-v2, rsync, Python, and iptables-persistent; creates a 2 GB /swapfile to guard builds against OOM; opens tcp/8080 in the host iptables INPUT chain before Oracle’s default trailing REJECT rule; creates ~/build.',
    checks: 'Verifies SSH connectivity and prints architecture (aarch64), core count (2), and memory (12 GB) before mutating the host; reminds the operator to restrict the VCN security list on tcp/8080 to Cloudflare IPv4 ranges only.',
  },
  {
    id: 'deploy',
    shortTitle: 'Build & Run',
    title: '2. Build & Deploy',
    scriptLabel: '02_deploy_alphazero.sh',
    summaryLine1: 'Sync src + 1 .pt,',
    summaryLine2: 'native aarch64 build',
    checkBadge: 'HTTP / (40 × 3 s)',
    command: 'bash deploy/alphazero-oracle/02_deploy_alphazero.sh',
    settings: 'DO_BUILD=true (or false), NO_CACHE=false, mem_limit: 4g',
    does: 'Rsyncs alphazero/ without .venv, runs/, models/, build/, cpp/build/, or *.so; SCPs one checkpoint as champion.pt; writes .env; builds infra/image/Dockerfile.prod natively on aarch64 (compiling C++ extensions and installing torch-cpu); starts gomoku-alphazero with restart: unless-stopped.',
    checks: 'Validates DEPLOY_ALPHAZERO_A1_HOST (IPv4), local config, and local checkpoint before SSH; polls http://127.0.0.1:8080/ up to 40 times at 3 s intervals (120 s max) while onnx-static exports ONNX and calibrates on 100 positions, dumping the last 30 container log lines on failure.',
  },
  {
    id: 'bench',
    shortTitle: 'Benchmark',
    title: '3. Search Budget',
    scriptLabel: '03_bench_search.sh',
    summaryLine1: '12 positions × 2.5 s',
    summaryLine2: 'in throwaway container',
    checkBadge: '613 → 671 sims',
    command: 'STOP_ENGINE=true bash deploy/alphazero-oracle/03_bench_search.sh',
    settings: 'ALPHAZERO_INFER_BACKEND=onnx-static, ALPHAZERO_THREADS=2, POSITIONS=12',
    does: 'Runs python -m gomoku.scripts.bench_search inside gomoku-alphazero:local with the exact served config, checkpoint, backend, and thread count. With STOP_ENGINE=true, stops the running container so no live traffic shares the 2 cores, then brings it back up.',
    checks: 'Reports median, min, max simulation counts and max elapsed time over 12 sampled positions under time_limit_s: 2.5. On A1, v5 iteration 144 measured 613 searches as trained and 671 after pruning dead channels.',
  },
  {
    id: 'smoke',
    shortTitle: 'Full Game',
    title: '4. Full-Game Check',
    scriptLabel: 'smoke_game.py',
    summaryLine1: 'SSH tunnel 18080→8080,',
    summaryLine2: 'self-play vs local rules',
    checkBadge: 'Rules + ≤ 3.0 s',
    command: 'ssh -N -L 18080:127.0.0.1:8080 ubuntu@<host> & python ../deploy/alphazero-oracle/smoke_game.py --url ws://127.0.0.1:18080/ws',
    settings: '--limit-s 3.0, --max-plies 200, optional --connections 2 --think-s 5',
    does: 'Connects to ws://127.0.0.1:18080/ws through an SSH tunnel (because port 8080 is closed to non-Cloudflare IPs) and sends frontend-formatted JSON move requests with recentMoves for every ply of a full game.',
    checks: 'Checks every move against a local Gomoku(use_native=False) rules engine: answer type == "move", move legality, exact board equality, exact capture scores, and round-trip latency <= 3.0 s. Baseline v5-pruned run averaged 2.52 s (max 2.59 s) with 262 MB peak container memory.',
  },
  {
    id: 'cutover',
    shortTitle: 'Cut Over',
    title: '5. Cloudflare Cut-Over',
    scriptLabel: 'cloudflare/deploy.sh',
    summaryLine1: 'DNS A (grey cloud) +',
    summaryLine2: 'Worker /alphazero/*',
    checkBadge: 'HTTPS + WSS check',
    command: 'bash deploy/cloudflare/deploy.sh',
    settings: 'DEPLOY_ALPHAZERO_IP=<A1 IPv4>, ALPHAZERO_ORIGIN=http://alphazero-api.<domain>:8080',
    does: 'Upserts DNS-only (proxied: false) A records for alphazero-api.<domain> and minimax-api.<domain>, and deploys the gomoku-ws-router Worker on /alphazero/*, /minimax/*, /gomoku, and /gomoku/* via wrangler.',
    checks: 'Automatically runs verify_connection.sh: verifies HTTP 200 {"status":"ok","model":"alphazero"} on https://<domain>/alphazero/, /gomoku proxy URL preservation, and WebSocket handshakes on wss://<domain>/alphazero/ws and /minimax/ws.',
  },
];

const currentStep = ref(0);
const activeStep = computed(() => STEPS[currentStep.value]);

function nodeFill(idx) {
  if (idx === currentStep.value) return '#eff6ff';
  if (idx < currentStep.value) return '#f0fdf4';
  return '#ffffff';
}

function nodeStroke(idx) {
  if (idx === currentStep.value) return '#2563eb';
  if (idx < currentStep.value) return '#16a34a';
  return '#cbd5e1';
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
  margin: 0 0 10px 0;
  color: #0f172a;
  letter-spacing: -0.2px;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 10px;
}

.ctrl-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 12px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.ctrl-btn:hover:not(:disabled) {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.ctrl-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.step-label {
  font-size: 13px;
  font-weight: 600;
  color: #475569;
  min-width: 82px;
  text-align: center;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.step-node {
  cursor: pointer;
}

.detail-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 10px;
  margin-top: 8px;
}

.detail-card {
  background: #ffffff;
  border: 1.5px solid #cbd5e1;
  border-radius: 8px;
  padding: 10px 12px;
}

.check-card {
  background: #f0fdf4;
  border-color: #86efac;
}

.detail-heading {
  font-size: 13px;
  font-weight: 700;
  color: #0f172a;
  margin-bottom: 4px;
}

.detail-text {
  margin: 0;
  font-size: 12.5px;
  line-height: 1.48;
  color: #334155;
}

.meta-bar {
  margin-top: 8px;
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: #475569;
  word-break: break-word;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
