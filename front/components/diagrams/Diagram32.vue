<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Self-Play Cluster Topology &amp; Inference Path</h1>

      <div class="controls-group">
        <div class="mode-buttons" role="group" aria-label="Self-play inference mode">
          <button
            type="button"
            class="mode-btn"
            :class="{ active: mode === 'remote' }"
            @click="mode = 'remote'"
          >
            Remote GPU Actors (pilot)
          </button>
          <button
            type="button"
            class="mode-btn"
            :class="{ active: mode === 'local' }"
            @click="mode = 'local'"
          >
            In-Worker bf16 + Head GPU Workers (v5 / v6)
          </button>
        </div>
      </div>

      <div class="status-bar" :class="mode === 'local' ? 'is-local' : 'is-remote'">
        <div class="status-main">
          <span class="badge" :class="mode === 'local' ? 'badge-green' : 'badge-amber'">
            {{ mode === 'local' ? 'use_local_inference: true' : 'use_local_inference: false' }}
          </span>
          <span class="status-text">
            <template v-if="mode === 'remote'">
              <strong>Remote actor path:</strong> CPU workers send batches via <code>ray.ObjectRef</code> to
              <code>12</code> GPU <code>RayInferenceActor</code>s on the head.
            </template>
            <template v-else>
              <strong>In-worker path:</strong> <code>0</code> inference actors; weights broadcast once per iteration, and each worker runs <code>LocalInference</code> in <code>bfloat16</code> synchronously.
            </template>
          </span>
        </div>
        <div class="status-sub">
          <template v-if="mode === 'remote'">
            Pilot (3 C4D + head): head capped at <strong>~770 req/s</strong> (~<strong>6,200 pos/s</strong>; ~6,000 in <code>v5-small-6x96.yaml</code>).
          </template>
          <template v-else>
            Pilot (3 C4D + head): <strong>~290 pos/s</strong> per C4D slot, <strong>~11,600 pos/s</strong> on 12 head GPU workers at batch 8 (~<strong>10,700 pos/s</strong> combined; 240 games in 15 min).
          </template>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Comparison between remote GPU inference actors and in-worker bfloat16 self-play inference"
      >
        <defs>
          <marker id="d32-arr-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#2563eb" />
          </marker>
          <marker id="d32-arr-amber" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#d97706" />
          </marker>
          <marker id="d32-arr-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#16a34a" />
          </marker>
          <marker id="d32-arr-slate" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#64748b" />
          </marker>
        </defs>

        <!-- Left Zone: CPU Worker Nodes -->
        <rect x="16" y="16" width="252" height="316" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="142" y="39" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
          CPU Worker VMs (Private, Cloud NAT)
        </text>
        <text x="142" y="56" text-anchor="middle" font-size="11.5" fill="#475569">
          {{ mode === 'remote' ? 'Pilot: 3 C4D (24 slots) · Full: 84 slots' : '9 C4D (72) + 2 C4 (12) = 84 slots' }}
        </text>

        <!-- RaySelfPlayWorker Box -->
        <rect x="30" y="72" width="224" height="172" rx="8" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5" />
        <text x="142" y="93" text-anchor="middle" font-size="12.5" font-weight="700" fill="#1e3a8a">
          RaySelfPlayWorker (1 CPU / slot)
        </text>

        <rect x="44" y="106" width="196" height="44" rx="6" fill="#ffffff" stroke="#60a5fa" stroke-width="1.2" />
        <text x="142" y="125" text-anchor="middle" font-size="12" font-weight="600" fill="#1e40af">
          C++ MctsEngine
        </text>
        <text x="142" y="141" text-anchor="middle" font-size="10.5" fill="#475569">
          {{ mode === 'remote' ? 'run_mcts_async_encoded (batch 8)' : 'run_mcts_batch_encoded (prefers_sync)' }}
        </text>

        <!-- Inference location inside CPU worker -->
        <rect
          x="44"
          y="164"
          width="196"
          height="66"
          rx="6"
          :fill="mode === 'local' ? '#dcfce7' : '#f8fafc'"
          :stroke="mode === 'local' ? '#16a34a' : '#cbd5e1'"
          :stroke-width="mode === 'local' ? '1.8' : '1.2'"
          :stroke-dasharray="mode === 'remote' ? '4 3' : undefined"
        />
        <text
          x="142"
          y="185"
          text-anchor="middle"
          font-size="12"
          font-weight="700"
          :fill="mode === 'local' ? '#166534' : '#64748b'"
        >
          {{ mode === 'local' ? 'LocalInference (CPU, bf16)' : 'RayInferenceClient' }}
        </text>
        <text x="142" y="202" text-anchor="middle" font-size="10.5" :fill="mode === 'local' ? '#15803d' : '#64748b'">
          {{ mode === 'local' ? 'torch.set_num_threads(1) · in-process' : 'Calls infer_async() over Ray' }}
        </text>
        <text x="142" y="218" text-anchor="middle" font-size="10.5" font-weight="600" :fill="mode === 'local' ? '#166534' : '#64748b'">
          {{ mode === 'local' ? '~290 pos/s per C4D slot (5 ms / pos)' : 'No local model copy on worker' }}
        </text>

        <!-- Arrow inside CPU worker -->
        <line
          x1="142"
          y1="150"
          x2="142"
          y2="162"
          :stroke="mode === 'local' ? '#16a34a' : '#2563eb'"
          stroke-width="1.8"
          :marker-end="mode === 'local' ? 'url(#d32-arr-green)' : 'url(#d32-arr-blue)'"
        />

        <!-- Throughput / Work-stealing note on CPU box -->
        <rect x="24" y="256" width="236" height="62" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />
        <text x="142" y="276" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">
          {{ mode === 'remote' ? 'Up-front split by worker count' : 'Dynamic chunk queue (2 queued / worker)' }}
        </text>
        <text x="142" y="293" text-anchor="middle" font-size="10" fill="#64748b">
          {{ mode === 'remote' ? 'All workers wait on head GPU RPC' : 'Faster GPU workers pull more games via ray.wait' }}
        </text>
        <text x="142" y="309" text-anchor="middle" font-size="10" fill="#64748b">
          {{ mode === 'remote' ? 'Slow replies flatten root visits if uncapped' : 'CPU + GPU workers finish together' }}
        </text>

        <!-- Center Bridge: Ray Object Store / Network -->
        <rect
          x="292"
          y="88"
          width="162"
          height="140"
          rx="8"
          :fill="mode === 'remote' ? '#fffbeb' : '#f1f5f9'"
          :stroke="mode === 'remote' ? '#d97706' : '#cbd5e1'"
          :stroke-width="mode === 'remote' ? '1.8' : '1.2'"
          :stroke-dasharray="mode === 'local' ? '5 4' : undefined"
        />
        <text
          x="373"
          y="113"
          text-anchor="middle"
          font-size="12"
          font-weight="700"
          :fill="mode === 'remote' ? '#92400e' : '#64748b'"
        >
          Ray Object Store
        </text>
        <text
          x="373"
          y="130"
          text-anchor="middle"
          font-size="11"
          :fill="mode === 'remote' ? '#b45309' : '#64748b'"
        >
          &amp; Network RPC
        </text>

        <template v-if="mode === 'remote'">
          <text x="373" y="156" text-anchor="middle" font-size="11" font-weight="600" fill="#92400e">
            Every batch (8 leaves)
          </text>
          <text x="373" y="173" text-anchor="middle" font-size="10.5" fill="#78350f">
            numpy → ObjectRef → actor
          </text>
          <text x="373" y="191" text-anchor="middle" font-size="10.5" fill="#78350f">
            returns (policy, value)
          </text>
          <text x="373" y="212" text-anchor="middle" font-size="11" font-weight="700" fill="#b45309">
            Bottleneck: ~770 req/s
          </text>
        </template>
        <template v-else>
          <text x="373" y="158" text-anchor="middle" font-size="11" font-weight="700" fill="#475569">
            Skipped for leaf evals
          </text>
          <text x="373" y="176" text-anchor="middle" font-size="10.5" fill="#64748b">
            Only weight sync once / iter
          </text>
          <text x="373" y="194" text-anchor="middle" font-size="10.5" fill="#64748b">
            and finished game records
          </text>
          <text x="373" y="213" text-anchor="middle" font-size="10.5" font-weight="600" fill="#15803d">
            Zero per-batch IPC
          </text>
        </template>

        <!-- Bridge Arrows -->
        <template v-if="mode === 'remote'">
          <line x1="254" y1="186" x2="290" y2="186" stroke="#d97706" stroke-width="2" marker-end="url(#d32-arr-amber)" />
          <line x1="292" y1="204" x2="256" y2="204" stroke="#d97706" stroke-width="1.8" stroke-dasharray="4 2" marker-end="url(#d32-arr-amber)" />
          <line x1="454" y1="186" x2="476" y2="186" stroke="#d97706" stroke-width="2" marker-end="url(#d32-arr-amber)" />
          <line x1="478" y1="204" x2="456" y2="204" stroke="#d97706" stroke-width="1.8" stroke-dasharray="4 2" marker-end="url(#d32-arr-amber)" />
        </template>
        <template v-else>
          <line x1="478" y1="138" x2="456" y2="138" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#d32-arr-slate)" />
          <line x1="292" y1="138" x2="256" y2="138" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#d32-arr-slate)" />
        </template>

        <!-- Right Zone: GPU Head Node -->
        <rect x="478" y="16" width="286" height="316" rx="10" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.5" />
        <text x="621" y="39" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
          GPU Head Node (g2-standard-16)
        </text>
        <text x="621" y="56" text-anchor="middle" font-size="11.5" fill="#475569">
          16 vCPU · 1 NVIDIA L4 GPU · External IP
        </text>

        <!-- Driver / Learner box -->
        <rect x="494" y="70" width="254" height="48" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.2" />
        <text x="621" y="90" text-anchor="middle" font-size="12" font-weight="700" fill="#1e293b">
          Driver &amp; Trainer (1 CPU + L4 after self-play)
        </text>
        <text x="621" y="106" text-anchor="middle" font-size="10.5" fill="#475569">
          Collects games → GCS Parquet → SGD/Adam on L4
        </text>

        <!-- GPU Actors / Head Workers -->
        <template v-if="mode === 'remote'">
          <rect x="494" y="132" width="254" height="122" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="1.8" />
          <text x="621" y="155" text-anchor="middle" font-size="12.5" font-weight="700" fill="#92400e">
            12 × RayInferenceActor
          </text>
          <text x="621" y="173" text-anchor="middle" font-size="11" fill="#78350f">
            actor_num_cpus: 1.0 · actor_num_gpus: 0.08
          </text>
          <text x="621" y="191" text-anchor="middle" font-size="11" fill="#78350f">
            Runs AsyncIO batching loop on L4 GPU
          </text>
          <rect x="510" y="204" width="222" height="36" rx="5" fill="#ffffff" stroke="#f59e0b" stroke-width="1.2" />
          <text x="621" y="220" text-anchor="middle" font-size="11" font-weight="700" fill="#92400e">
            Measured pilot total: ~6,200 pos/s
          </text>
          <text x="621" y="234" text-anchor="middle" font-size="10" fill="#78350f">
            Head saturates around ~770 req/s
          </text>
        </template>
        <template v-else>
          <rect x="494" y="132" width="254" height="122" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.8" />
          <text x="621" y="155" text-anchor="middle" font-size="12.5" font-weight="700" fill="#166534">
            13 × GPU RaySelfPlayWorker
          </text>
          <text x="621" y="173" text-anchor="middle" font-size="11" fill="#15803d">
            num_actors: 0 (no inference actors)
          </text>
          <text x="621" y="191" text-anchor="middle" font-size="11" fill="#15803d">
            13 workers share 0.95 L4 (bf16 in-process)
          </text>
          <rect x="510" y="204" width="222" height="36" rx="5" fill="#ffffff" stroke="#16a34a" stroke-width="1.2" />
          <text x="621" y="220" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">
            12 GPU workers (pilot): ~11,600 pos/s
          </text>
          <text x="621" y="234" text-anchor="middle" font-size="10" fill="#15803d">
            Freed inference actor CPU → 13th worker
          </text>
        </template>

        <!-- Head GPU Share Footer -->
        <rect x="494" y="266" width="254" height="52" rx="6" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1.2" />
        <text x="621" y="286" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">
          {{ mode === 'remote' ? 'Head CPU allocation (pilot):' : 'Head CPU allocation (v5-cont / v6):' }}
        </text>
        <text x="621" y="304" text-anchor="middle" font-size="10.5" fill="#64748b">
          {{ mode === 'remote' ? '12 inference actors + driver (no head self-play)' : '13 GPU self-play workers + driver (84 + 13 = 97 total)' }}
        </text>

        <!-- Bottom Summary Strip -->
        <rect x="16" y="344" width="748" height="36" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />
        <text x="390" y="366" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">
          <template v-if="mode === 'remote'">
            Remote path: 1 network hop per 8 leaves · ~6,000–6,200 pos/s on pilot · requires max_inflight_batches cap
          </template>
          <template v-else>
            In-worker bf16 path: 0 network hops per search · ~10,700 pos/s on pilot · 97 concurrent self-play workers at full scale
          </template>
        </text>
      </svg>

      <p class="source-note">
        Sources: <code>v5-small-6x96-pilot.yaml</code>, <code>v5-small-6x96-local-pilot.yaml</code>, <code>v5-small-6x96.yaml</code>, <code>v5-small-6x96-cont.yaml</code>, <code>ray_runner.py</code>, and <code>local.py</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const mode = ref('local');
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

.controls-group {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}

.mode-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.mode-btn {
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

.mode-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.mode-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.status-bar {
  background: #ffffff;
  border: 1.5px solid #e2e8f0;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.45;
  color: #334155;
}

.status-bar.is-local {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-remote {
  background: #fffbeb;
  border-color: #fcd34d;
}

.status-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.badge {
  display: inline-block;
  font-size: 11.5px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 999px;
}

.badge-green {
  background: #dcfce7;
  color: #166534;
  border: 1px solid #86efac;
}

.badge-amber {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fcd34d;
}

.status-sub {
  font-size: 12.5px;
  color: #475569;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.source-note {
  margin: 8px 4px 0 4px;
  font-size: 12px;
  line-height: 1.45;
  color: #64748b;
  text-align: center;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
