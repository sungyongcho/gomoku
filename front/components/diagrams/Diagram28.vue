<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Cloudflare Worker Routing &amp; Origin Security-List Lockdown</h1>

      <div class="controls-group">
        <div class="mode-buttons" role="group" aria-label="Traffic path scenario">
          <button
            v-for="m in MODES"
            :key="m.id"
            type="button"
            class="mode-btn"
            :class="{ active: activeModeId === m.id, 'is-danger': m.id === 'direct' && activeModeId === m.id }"
            @click="selectMode(m.id)"
          >
            {{ m.label }}
          </button>
        </div>
        <button type="button" class="anim-btn" @click="togglePlaying">
          {{ isPlaying ? '⏸ Pause packet' : '▶ Animate packet' }}
        </button>
      </div>

      <div class="status-bar" :class="statusBarClass">
        <div class="status-main">
          <span class="badge" :class="badgeClass">{{ activeMode.badge }}</span>
          <span class="status-title"><strong>{{ activeMode.title }}</strong></span>
        </div>
        <div class="status-desc">{{ activeMode.description }}</div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 760 285"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Network path from browser through Cloudflare Worker to Oracle A1 origin port 8080, showing direct access refused by the VCN security list"
      >
        <defs>
          <marker id="d28-arr-green" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#16a34a" />
          </marker>
          <marker id="d28-arr-red" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#dc2626" />
          </marker>
          <marker id="d28-arr-blue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#2563eb" />
          </marker>
          <marker id="d28-arr-dim" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6 Z" fill="#cbd5e1" />
          </marker>
        </defs>

        <!-- Oracle Cloud boundary box -->
        <rect
          x="398"
          y="14"
          width="348"
          height="248"
          rx="10"
          fill="#f1f5f9"
          stroke="#94a3b8"
          stroke-width="1.5"
          stroke-dasharray="5 4"
        />
        <text x="572" y="34" text-anchor="middle" font-size="12" font-weight="700" fill="#475569">
          Oracle Cloud VCN &amp; A1 Instance (origin)
        </text>

        <!-- Node 1: Browser / Client (top left) -->
        <rect
          x="14"
          y="48"
          width="138"
          height="68"
          rx="8"
          :fill="activeModeId === 'worker' ? '#eff6ff' : '#ffffff'"
          :stroke="activeModeId === 'worker' ? '#2563eb' : '#cbd5e1'"
          stroke-width="2"
        />
        <text x="83" y="72" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">
          Player Browser
        </text>
        <text x="83" y="90" text-anchor="middle" font-size="11" fill="#475569">
          wss://&lt;domain&gt;
        </text>
        <text x="83" y="105" text-anchor="middle" font-size="11" font-weight="600" fill="#1e40af">
          /alphazero/ws
        </text>

        <!-- Node 2: Direct Non-Cloudflare / Operator Workstation (bottom left) -->
        <rect
          x="14"
          y="168"
          width="138"
          height="74"
          rx="8"
          :fill="activeModeId === 'direct' ? '#fef2f2' : activeModeId === 'tunnel' ? '#eff6ff' : '#ffffff'"
          :stroke="activeModeId === 'direct' ? '#dc2626' : activeModeId === 'tunnel' ? '#2563eb' : '#cbd5e1'"
          stroke-width="2"
        />
        <text x="83" y="191" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">
          {{ activeModeId === 'tunnel' ? 'Operator Host' : 'Direct Caller' }}
        </text>
        <text x="83" y="209" text-anchor="middle" font-size="11" fill="#475569">
          {{ activeModeId === 'tunnel' ? 'smoke_game.py' : 'Non-Cloudflare IP' }}
        </text>
        <text
          x="83"
          y="226"
          text-anchor="middle"
          font-size="10"
          font-weight="600"
          :fill="activeModeId === 'direct' ? '#b91c1c' : '#1e40af'"
        >
          {{ activeModeId === 'tunnel' ? 'ssh -L 18080:127.0.0.1:8080' : 'http://origin:8080' }}
        </text>

        <!-- Node 3: Cloudflare Worker (top center) -->
        <rect
          x="190"
          y="42"
          width="190"
          height="82"
          rx="8"
          :fill="activeModeId === 'worker' ? '#fff7ed' : '#ffffff'"
          :stroke="activeModeId === 'worker' ? '#ea580c' : '#cbd5e1'"
          stroke-width="2"
        />
        <text x="285" y="64" text-anchor="middle" font-size="13" font-weight="700" fill="#9a3412">
          Cloudflare Worker
        </text>
        <text x="285" y="81" text-anchor="middle" font-size="10.5" fill="#475569">
          TLS ends here · strip /alphazero
        </text>
        <text x="285" y="97" text-anchor="middle" font-size="10" font-weight="600" fill="#9a3412">
          → alphazero-api.&lt;domain&gt;:8080
        </text>
        <text x="285" y="113" text-anchor="middle" font-size="10" fill="#64748b">
          Egress: 15 Cloudflare IPv4 CIDRs
        </text>

        <!-- Node 4: VCN Security List (firewall gate) -->
        <rect
          x="412"
          y="48"
          width="144"
          height="196"
          rx="8"
          :fill="activeModeId === 'direct' ? '#fef2f2' : '#f0fdf4'"
          :stroke="activeModeId === 'direct' ? '#dc2626' : '#16a34a'"
          stroke-width="2"
        />
        <text x="484" y="70" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">
          VCN Security List
        </text>
        <line x1="424" y1="80" x2="544" y2="80" stroke="#cbd5e1" stroke-width="1" />

        <!-- Rule 1: port 8080 -->
        <rect
          x="422"
          y="90"
          width="124"
          height="58"
          rx="6"
          :fill="activeModeId === 'worker' ? '#dcfce7' : activeModeId === 'direct' ? '#fee2e2' : '#ffffff'"
          :stroke="activeModeId === 'worker' ? '#16a34a' : activeModeId === 'direct' ? '#dc2626' : '#cbd5e1'"
          stroke-width="1.5"
        />
        <text x="484" y="109" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          tcp/8080 ingress
        </text>
        <text x="484" y="124" text-anchor="middle" font-size="10.5" fill="#334155">
          Only Cloudflare IPv4
        </text>
        <text x="484" y="139" text-anchor="middle" font-size="10.5" font-weight="600" :fill="activeModeId === 'direct' ? '#b91c1c' : '#15803d'">
          {{ activeModeId === 'direct' ? '✗ Non-CF dropped' : '✓ 15 CIDRs allowed' }}
        </text>

        <!-- Rule 2: port 22 -->
        <rect
          x="422"
          y="164"
          width="124"
          height="58"
          rx="6"
          :fill="activeModeId === 'tunnel' ? '#dbeafe' : '#ffffff'"
          :stroke="activeModeId === 'tunnel' ? '#2563eb' : '#cbd5e1'"
          stroke-width="1.5"
        />
        <text x="484" y="183" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          tcp/22 ingress
        </text>
        <text x="484" y="198" text-anchor="middle" font-size="10.5" fill="#334155">
          Operator IP (/32)
        </text>
        <text x="484" y="213" text-anchor="middle" font-size="10.5" font-weight="600" fill="#1e40af">
          {{ activeModeId === 'tunnel' ? '✓ Tunnel admitted' : 'SSH &amp; checks only' }}
        </text>

        <!-- Node 5: Host iptables + Docker Container (right) -->
        <rect
          x="582"
          y="48"
          width="150"
          height="196"
          rx="8"
          fill="#ffffff"
          :stroke="activeModeId === 'direct' ? '#cbd5e1' : '#16a34a'"
          stroke-width="2"
        />
        <text x="657" y="70" text-anchor="middle" font-size="12.5" font-weight="700" fill="#0f172a">
          A1 Host &amp; Engine
        </text>
        <text x="657" y="88" text-anchor="middle" font-size="10.5" fill="#64748b">
          iptables INPUT ACCEPT :8080
        </text>

        <rect x="594" y="102" width="126" height="126" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5" />
        <text x="657" y="122" text-anchor="middle" font-size="11.5" font-weight="700" fill="#0f172a">
          gomoku-alphazero
        </text>
        <text x="657" y="140" text-anchor="middle" font-size="10.5" fill="#475569">
          uvicorn :8080 (plain WS)
        </text>
        <text x="657" y="158" text-anchor="middle" font-size="10.5" fill="#475569">
          2 Neoverse-N1 cores
        </text>
        <text x="657" y="176" text-anchor="middle" font-size="10.5" fill="#475569">
          static int8 · mem_limit 4g
        </text>
        <text x="657" y="198" text-anchor="middle" font-size="10.5" font-weight="600" fill="#15803d">
          /ws · / (health)
        </text>
        <text x="657" y="216" text-anchor="middle" font-size="10" fill="#64748b">
          2.5 s limit (&lt; 3 s target)
        </text>

        <!-- Path arrows -->
        <!-- Browser -> Worker -->
        <line
          x1="152"
          y1="82"
          x2="186"
          y2="82"
          :stroke="activeModeId === 'worker' ? '#16a34a' : '#cbd5e1'"
          :stroke-width="activeModeId === 'worker' ? '2.5' : '1.5'"
          :marker-end="activeModeId === 'worker' ? 'url(#d28-arr-green)' : 'url(#d28-arr-dim)'"
        />
        <!-- Worker -> Security List :8080 -->
        <line
          x1="380"
          y1="108"
          x2="418"
          y2="108"
          :stroke="activeModeId === 'worker' ? '#16a34a' : '#cbd5e1'"
          :stroke-width="activeModeId === 'worker' ? '2.5' : '1.5'"
          :marker-end="activeModeId === 'worker' ? 'url(#d28-arr-green)' : 'url(#d28-arr-dim)'"
        />
        <!-- Security List :8080 -> Container -->
        <line
          x1="546"
          y1="118"
          x2="588"
          y2="118"
          :stroke="activeModeId === 'worker' ? '#16a34a' : '#cbd5e1'"
          :stroke-width="activeModeId === 'worker' ? '2.5' : '1.5'"
          :marker-end="activeModeId === 'worker' ? 'url(#d28-arr-green)' : 'url(#d28-arr-dim)'"
        />

        <!-- Direct Caller -> Security List (blocked at x=412) -->
        <path
          d="M 152,195 L 390,195 L 390,132 L 414,132"
          fill="none"
          :stroke="activeModeId === 'direct' ? '#dc2626' : '#e2e8f0'"
          :stroke-width="activeModeId === 'direct' ? '2.5' : '1.2'"
          :stroke-dasharray="activeModeId === 'direct' ? '5 3' : 'none'"
          :marker-end="activeModeId === 'direct' ? 'url(#d28-arr-red)' : 'url(#d28-arr-dim)'"
        />
        <g v-if="activeModeId === 'direct'">
          <circle cx="404" cy="132" r="10" fill="#dc2626" />
          <text x="404" y="136" text-anchor="middle" font-size="12" font-weight="800" fill="#ffffff">✕</text>
          <rect x="200" y="156" width="172" height="28" rx="5" fill="#fef2f2" stroke="#fca5a5" stroke-width="1" />
          <text x="286" y="174" text-anchor="middle" font-size="11" font-weight="700" fill="#b91c1c">
            Blocked before host / engine
          </text>
        </g>

        <!-- Operator -> SSH tcp/22 -> Loopback Container -->
        <line
          x1="152"
          y1="205"
          x2="418"
          y2="205"
          :stroke="activeModeId === 'tunnel' ? '#2563eb' : '#e2e8f0'"
          :stroke-width="activeModeId === 'tunnel' ? '2.5' : '1.2'"
          :marker-end="activeModeId === 'tunnel' ? 'url(#d28-arr-blue)' : 'url(#d28-arr-dim)'"
        />
        <line
          x1="546"
          y1="195"
          x2="588"
          y2="195"
          :stroke="activeModeId === 'tunnel' ? '#2563eb' : '#e2e8f0'"
          :stroke-width="activeModeId === 'tunnel' ? '2.5' : '1.2'"
          :marker-end="activeModeId === 'tunnel' ? 'url(#d28-arr-blue)' : 'url(#d28-arr-dim)'"
        />

        <!-- Animated packet circle -->
        <circle
          :cx="packetPos.x"
          :cy="packetPos.y"
          r="6.5"
          :fill="packetColor"
          stroke="#ffffff"
          stroke-width="2"
        />

        <!-- Footer inside SVG -->
        <text x="380" y="276" text-anchor="middle" font-size="11.5" fill="#64748b">
          Packet motion is illustrative. Security-list rules: tcp/22 from operator IP, tcp/8080 from Cloudflare IPv4 only.
        </text>
      </svg>

      <p class="source-note">
        Sources: <code>deploy/cloudflare/cloudflare-worker.js</code>, <code>deploy/cloudflare/wrangler.toml</code>, <code>deploy/alphazero-oracle/README.md</code>, and <code>deploy/alphazero-oracle/docker-compose.yml</code>.
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

const MODES = [
  {
    id: 'worker',
    label: '1. Via Cloudflare Worker',
    badge: 'Allowed (HTTPS / WSS)',
    title: 'Browser → Cloudflare Worker → Origin tcp/8080',
    description:
      'TLS terminates at Cloudflare. The Worker matches /alphazero/*, strips /alphazero, sets Host to alphazero-api.<domain>:8080, and connects from one of Cloudflare’s 15 IPv4 CIDRs—admitted by the VCN security list and host iptables to uvicorn on port 8080.',
    points: [
      { x: 152, y: 82 },
      { x: 190, y: 82 },
      { x: 380, y: 108 },
      { x: 422, y: 108 },
      { x: 546, y: 118 },
      { x: 594, y: 118 },
    ],
  },
  {
    id: 'direct',
    label: '2. Direct to Origin :8080',
    badge: 'Refused at VCN Security List',
    title: 'Non-Cloudflare IP → Origin tcp/8080 (Dropped)',
    description:
      'A direct request to <origin>:8080 from outside Cloudflare’s 15 IPv4 ranges is dropped at the Oracle VCN security list before reaching iptables or the container. Because port 8080 serves plain HTTP/WS with no TLS or auth and only 2 CPU cores exist, bypassing the proxy is blocked.',
    points: [
      { x: 152, y: 195 },
      { x: 390, y: 195 },
      { x: 390, y: 132 },
      { x: 404, y: 132 },
    ],
  },
  {
    id: 'tunnel',
    label: '3. Operator SSH Tunnel',
    badge: 'Allowed (Operator /32 on tcp/22)',
    title: 'smoke_game.py → SSH Tunnel (tcp/22) → 127.0.0.1:8080',
    description:
      'To test a full game without opening port 8080 to the public internet, smoke_game.py connects through ssh -N -L 18080:127.0.0.1:8080 over tcp/22 (admitted from the operator’s IP).',
    points: [
      { x: 152, y: 205 },
      { x: 422, y: 205 },
      { x: 546, y: 195 },
      { x: 594, y: 195 },
    ],
  },
];

const activeModeId = ref('worker');
const progress = ref(0.5);
const isPlaying = ref(true);
let timerId = null;

const activeMode = computed(
  () => MODES.find((m) => m.id === activeModeId.value) ?? MODES[0]
);

const statusBarClass = computed(() => {
  if (activeModeId.value === 'direct') return 'is-danger';
  if (activeModeId.value === 'tunnel') return 'is-info';
  return 'is-ok';
});

const badgeClass = computed(() => {
  if (activeModeId.value === 'direct') return 'badge-red';
  if (activeModeId.value === 'tunnel') return 'badge-blue';
  return 'badge-green';
});

const packetColor = computed(() => {
  if (activeModeId.value === 'direct') return '#dc2626';
  if (activeModeId.value === 'tunnel') return '#2563eb';
  return '#16a34a';
});

const packetPos = computed(() => {
  const pts = activeMode.value.points;
  if (pts.length < 2) return pts[0] ?? { x: 0, y: 0 };

  const segLens = [];
  let total = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const dx = pts[i + 1].x - pts[i].x;
    const dy = pts[i + 1].y - pts[i].y;
    const len = Math.hypot(dx, dy);
    segLens.push(len);
    total += len;
  }
  if (total === 0) return pts[0];

  let target = (progress.value % 1) * total;
  for (let i = 0; i < segLens.length; i++) {
    if (target <= segLens[i] || i === segLens.length - 1) {
      const t = segLens[i] > 0 ? target / segLens[i] : 0;
      return {
        x: pts[i].x + (pts[i + 1].x - pts[i].x) * t,
        y: pts[i].y + (pts[i + 1].y - pts[i].y) * t,
      };
    }
    target -= segLens[i];
  }
  return pts[pts.length - 1];
});

function selectMode(id) {
  activeModeId.value = id;
  progress.value = 0.0;
}

function startTimer() {
  if (timerId !== null) return;
  timerId = setInterval(() => {
    progress.value = (progress.value + 0.02) % 1;
  }, 45);
}

function stopTimer() {
  if (timerId !== null) {
    clearInterval(timerId);
    timerId = null;
  }
}

function togglePlaying() {
  isPlaying.value = !isPlaying.value;
  if (isPlaying.value) {
    startTimer();
  } else {
    stopTimer();
  }
}

onMounted(() => {
  if (isPlaying.value) {
    startTimer();
  }
});

onBeforeUnmount(() => {
  stopTimer();
});
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
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 10px;
}

.mode-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.mode-btn,
.anim-btn {
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

.mode-btn:hover,
.anim-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.mode-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.mode-btn.active.is-danger {
  background: #fef2f2;
  border-color: #dc2626;
  color: #b91c1c;
}

.status-bar {
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 10px;
  font-size: 13px;
  line-height: 1.45;
  color: #334155;
  border: 1.5px solid #e2e8f0;
  background: #ffffff;
}

.status-bar.is-ok {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-danger {
  background: #fef2f2;
  border-color: #fca5a5;
}

.status-bar.is-info {
  background: #eff6ff;
  border-color: #93c5fd;
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

.badge-red {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fca5a5;
}

.badge-blue {
  background: #dbeafe;
  color: #1e40af;
  border: 1px solid #93c5fd;
}

.status-desc {
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
