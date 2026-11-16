<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>Filling the 100-vCPU GCP Quota Across Cluster Revisions</h1>

      <div class="controls-group">
        <div class="stage-buttons" role="group" aria-label="Cluster sizing revisions">
          <button
            v-for="stage in STAGES"
            :key="stage.id"
            type="button"
            class="stage-btn"
            :class="{ active: stageId === stage.id }"
            @click="stageId = stage.id"
          >
            {{ stage.shortLabel }}
          </button>
        </div>
      </div>

      <div class="status-bar" :class="activeStage.statusClass">
        <div class="status-main">
          <span class="badge" :class="activeStage.badgeClass">{{ activeStage.commit }}</span>
          <span class="status-text">
            <strong>{{ activeStage.title }}:</strong> {{ activeStage.summary }}
          </span>
        </div>
        <div class="status-sub">
          Active vCPUs: <strong>{{ activeStage.activeVcpus }} / 100</strong> ·
          C4D vCPUs: <strong>{{ activeStage.c4dActive }} / 72</strong> ·
          Self-play slots: <strong>{{ activeStage.cpuSlots }} CPU</strong>
          <template v-if="activeStage.gpuWorkers > 0">
            + <strong>{{ activeStage.gpuWorkers }} GPU</strong> = <strong>{{ activeStage.cpuSlots + activeStage.gpuWorkers }} total</strong>
          </template>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 390"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="How GCP machine counts fill the 72-vCPU C4D quota and 100-vCPU all-regions quota"
      >
        <!-- Bar 1: All-regions 100-vCPU quota -->
        <text x="20" y="26" font-size="13.5" font-weight="700" fill="#0f172a">
          All-Regions CPU Quota (100 vCPUs limit)
        </text>
        <text x="760" y="26" text-anchor="end" font-size="12" font-weight="600" fill="#475569">
          {{ activeStage.requestedVcpus }} vCPUs requested ({{ activeStage.activeVcpus }} running)
        </text>

        <!-- Track (0 to 108 scale) -->
        <rect x="20" y="38" width="740" height="36" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" />

        <!-- Stacked segments -->
        <rect
          x="20"
          y="38"
          :width="vcpuW(16)"
          height="36"
          rx="4"
          fill="#2563eb"
        />
        <text :x="20 + vcpuW(16) / 2" y="60" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">
          Head 16
        </text>

        <rect
          :x="20 + vcpuW(16)"
          y="38"
          :width="vcpuW(activeStage.c4dActive)"
          height="36"
          fill="#16a34a"
        />
        <text
          :x="20 + vcpuW(16) + vcpuW(activeStage.c4dActive) / 2"
          y="60"
          text-anchor="middle"
          font-size="11.5"
          font-weight="700"
          fill="#ffffff"
        >
          C4D {{ activeStage.c4dActive }} vCPU
        </text>

        <rect
          v-if="activeStage.c4Vcpus > 0"
          :x="20 + vcpuW(16 + activeStage.c4dActive)"
          y="38"
          :width="vcpuW(activeStage.c4Vcpus)"
          height="36"
          fill="#7c3aed"
        />
        <text
          v-if="activeStage.c4Vcpus > 0"
          :x="20 + vcpuW(16 + activeStage.c4dActive) + vcpuW(activeStage.c4Vcpus) / 2"
          y="60"
          text-anchor="middle"
          font-size="11"
          font-weight="700"
          fill="#ffffff"
        >
          C4 {{ activeStage.c4Vcpus }}
        </text>

        <!-- Unstarted over-quota segment -->
        <rect
          v-if="activeStage.blockedVcpus > 0"
          :x="20 + vcpuW(activeStage.activeVcpus)"
          y="38"
          :width="vcpuW(activeStage.blockedVcpus)"
          height="36"
          rx="3"
          fill="#fecaca"
          stroke="#dc2626"
          stroke-width="1.5"
          stroke-dasharray="4 3"
        />
        <text
          v-if="activeStage.blockedVcpus > 0"
          :x="20 + vcpuW(activeStage.activeVcpus) + vcpuW(activeStage.blockedVcpus) / 2"
          y="60"
          text-anchor="middle"
          font-size="11"
          font-weight="700"
          fill="#991b1b"
        >
          Blocked +{{ activeStage.blockedVcpus }}
        </text>

        <!-- 100 vCPU Quota Guide Line -->
        <line
          :x1="20 + vcpuW(100)"
          y1="32"
          :x2="20 + vcpuW(100)"
          y2="82"
          stroke="#dc2626"
          stroke-width="2"
          stroke-dasharray="4 3"
        />
        <text :x="20 + vcpuW(100) - 4" y="96" text-anchor="end" font-size="11" font-weight="700" fill="#dc2626">
          100-vCPU ceiling
        </text>

        <!-- Bar 2: C4D Regional Quota (72 vCPUs) -->
        <text x="20" y="122" font-size="13.5" font-weight="700" fill="#0f172a">
          Regional C4D (AMD Turin) Quota (72 vCPUs limit)
        </text>
        <rect x="20" y="132" width="740" height="28" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" />
        <rect x="20" y="132" :width="vcpuW(activeStage.c4dActive)" height="28" rx="4" fill="#16a34a" />
        <text :x="20 + vcpuW(activeStage.c4dActive) / 2" y="150" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">
          {{ activeStage.c4dActive / 8 }} × c4d-standard-8 = {{ activeStage.c4dActive }} vCPU
        </text>
        <rect
          v-if="activeStage.blockedVcpus > 0"
          :x="20 + vcpuW(activeStage.c4dActive)"
          y="132"
          :width="vcpuW(activeStage.blockedVcpus)"
          height="28"
          rx="3"
          fill="#fecaca"
          stroke="#dc2626"
          stroke-width="1.5"
          stroke-dasharray="4 3"
        />
        <text
          v-if="activeStage.blockedVcpus > 0"
          :x="20 + vcpuW(activeStage.c4dActive) + vcpuW(activeStage.blockedVcpus) / 2"
          y="150"
          text-anchor="middle"
          font-size="11"
          font-weight="700"
          fill="#991b1b"
        >
          2 blocked (+16)
        </text>
        <line
          :x1="20 + vcpuW(72)"
          y1="126"
          :x2="20 + vcpuW(72)"
          y2="166"
          stroke="#dc2626"
          stroke-width="2"
          stroke-dasharray="4 3"
        />
        <text :x="20 + vcpuW(72) - 4" y="178" text-anchor="end" font-size="11" font-weight="700" fill="#dc2626">
          72-vCPU C4D limit
        </text>

        <!-- Divider -->
        <line x1="20" y1="194" x2="760" y2="194" stroke="#e2e8f0" stroke-width="1.5" />

        <!-- Machine Breakdown Cards -->
        <text x="20" y="216" font-size="13.5" font-weight="700" fill="#0f172a">
          Node Types &amp; Self-Play Worker Allocation
        </text>

        <g v-for="(card, idx) in activeStage.cards" :key="card.name">
          <rect
            :x="20 + idx * 188"
            y="228"
            width="176"
            height="146"
            rx="8"
            :fill="card.fill"
            :stroke="card.stroke"
            stroke-width="1.6"
            :stroke-dasharray="card.dashed ? '5 3' : undefined"
          />
          <text :x="108 + idx * 188" y="250" text-anchor="middle" font-size="12.5" font-weight="700" :fill="card.titleColor">
            {{ card.name }}
          </text>
          <text :x="108 + idx * 188" y="268" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">
            {{ card.machine }}
          </text>
          <line :x1="34 + idx * 188" y1="278" :x2="182 + idx * 188" y2="278" stroke="#cbd5e1" stroke-width="1" />
          <text :x="108 + idx * 188" y="298" text-anchor="middle" font-size="13.5" font-weight="700" fill="#0f172a">
            {{ card.countLabel }}
          </text>
          <text :x="108 + idx * 188" y="317" text-anchor="middle" font-size="11.5" fill="#475569">
            {{ card.vcpuLabel }}
          </text>
          <text :x="108 + idx * 188" y="338" text-anchor="middle" font-size="11.5" font-weight="700" :fill="card.slotColor">
            {{ card.slotLabel }}
          </text>
          <text :x="108 + idx * 188" y="357" text-anchor="middle" font-size="10.5" fill="#64748b">
            {{ card.note }}
          </text>
        </g>
      </svg>

      <p class="source-note">
        Sources: <code>cluster_v5-pilot.yaml</code> (<code>eaf7fa7</code>), <code>cluster_v5.yaml</code> (<code>eaf7fa7</code>, <code>47e5258</code>, <code>aaa462f</code>), and <code>v5-small-6x96-cont.yaml</code> (<code>b90e04e</code>).
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const STAGES = [
  {
    id: 'pilot',
    shortLabel: 'Pilot (eaf7fa7)',
    commit: 'eaf7fa7 · cluster_v5-pilot.yaml',
    title: '3-Worker Pilot',
    summary: 'Small cluster to validate GPU inference, GCS replay storage, resume, and throughput.',
    statusClass: '',
    badgeClass: 'badge-blue',
    activeVcpus: 40,
    requestedVcpus: 40,
    c4dActive: 24,
    c4Vcpus: 0,
    blockedVcpus: 0,
    cpuSlots: 24,
    gpuWorkers: 0,
    cards: [
      {
        name: 'GPU Head',
        machine: 'g2-standard-16 + 1 L4',
        countLabel: '1 VM',
        vcpuLabel: '16 vCPUs',
        slotLabel: '12 inference actors',
        note: 'Pilot remote inference path',
        fill: '#eff6ff',
        stroke: '#2563eb',
        titleColor: '#1e3a8a',
        slotColor: '#1e40af',
        dashed: false,
      },
      {
        name: 'AMD Turin Workers',
        machine: 'c4d-standard-8',
        countLabel: '3 VMs',
        vcpuLabel: '24 vCPUs (3 × 8)',
        slotLabel: '24 CPU self-play slots',
        note: 'Private (Cloud NAT)',
        fill: '#f0fdf4',
        stroke: '#16a34a',
        titleColor: '#166534',
        slotColor: '#15803d',
        dashed: false,
      },
      {
        name: 'Intel C4 Workers',
        machine: 'not used in pilot',
        countLabel: '0 VMs',
        vcpuLabel: '0 vCPUs',
        slotLabel: '0 slots',
        note: 'Added later in aaa462f',
        fill: '#f8fafc',
        stroke: '#cbd5e1',
        titleColor: '#64748b',
        slotColor: '#64748b',
        dashed: true,
      },
      {
        name: 'Unused Quota',
        machine: 'Remaining under 100',
        countLabel: '60 vCPUs free',
        vcpuLabel: '40 / 100 used',
        slotLabel: '24 total self-play',
        note: 'Checked before full scale',
        fill: '#f8fafc',
        stroke: '#cbd5e1',
        titleColor: '#475569',
        slotColor: '#334155',
        dashed: false,
      },
    ],
  },
  {
    id: 'initial',
    shortLabel: 'Initial v5 (eaf7fa7)',
    commit: 'eaf7fa7 · cluster_v5.yaml',
    title: 'Initial 9-Worker v5',
    summary: 'Filled the 72-vCPU C4D quota with 9 workers, leaving 12 vCPUs unused under the 100-vCPU limit.',
    statusClass: '',
    badgeClass: 'badge-blue',
    activeVcpus: 88,
    requestedVcpus: 88,
    c4dActive: 72,
    c4Vcpus: 0,
    blockedVcpus: 0,
    cpuSlots: 72,
    gpuWorkers: 0,
    cards: [
      {
        name: 'GPU Head',
        machine: 'g2-standard-16 + 1 L4',
        countLabel: '1 VM',
        vcpuLabel: '16 vCPUs',
        slotLabel: 'Remote inference actors',
        note: 'Before local bf16 switch',
        fill: '#eff6ff',
        stroke: '#2563eb',
        titleColor: '#1e3a8a',
        slotColor: '#1e40af',
        dashed: false,
      },
      {
        name: 'AMD Turin Workers',
        machine: 'c4d-standard-8',
        countLabel: '9 VMs',
        vcpuLabel: '72 vCPUs (9 × 8)',
        slotLabel: '72 CPU self-play slots',
        note: 'Hits 72-vCPU C4D cap',
        fill: '#f0fdf4',
        stroke: '#16a34a',
        titleColor: '#166534',
        slotColor: '#15803d',
        dashed: false,
      },
      {
        name: 'Intel C4 Workers',
        machine: 'not added yet',
        countLabel: '0 VMs',
        vcpuLabel: '0 vCPUs',
        slotLabel: '0 slots',
        note: '12 vCPUs left unallocated',
        fill: '#f8fafc',
        stroke: '#cbd5e1',
        titleColor: '#64748b',
        slotColor: '#64748b',
        dashed: true,
      },
      {
        name: 'Remaining Quota',
        machine: 'Under 100-vCPU cap',
        countLabel: '12 vCPUs idle',
        vcpuLabel: '88 / 100 used',
        slotLabel: '72 total self-play',
        note: 'ray_local_num_workers: 72',
        fill: '#f8fafc',
        stroke: '#cbd5e1',
        titleColor: '#475569',
        slotColor: '#334155',
        dashed: false,
      },
    ],
  },
  {
    id: 'demand',
    shortLabel: 'Quota Demand (47e5258)',
    commit: '47e5258 · cluster_v5.yaml',
    title: 'Requesting 11 C4D Workers',
    summary: 'Asked for 11 C4D workers (104 vCPUs total) so the 2 blocked VMs would log quota demand while 9 ran.',
    statusClass: 'is-warn',
    badgeClass: 'badge-amber',
    activeVcpus: 88,
    requestedVcpus: 104,
    c4dActive: 72,
    c4Vcpus: 0,
    blockedVcpus: 16,
    cpuSlots: 72,
    gpuWorkers: 12,
    cards: [
      {
        name: 'GPU Head',
        machine: 'g2-standard-16 + 1 L4',
        countLabel: '1 VM',
        vcpuLabel: '16 vCPUs',
        slotLabel: '12 GPU self-play workers',
        note: '+1 unused actor + driver',
        fill: '#eff6ff',
        stroke: '#2563eb',
        titleColor: '#1e3a8a',
        slotColor: '#1e40af',
        dashed: false,
      },
      {
        name: 'AMD Turin (Active)',
        machine: 'c4d-standard-8',
        countLabel: '9 VMs started',
        vcpuLabel: '72 vCPUs (9 × 8)',
        slotLabel: '72 CPU self-play slots',
        note: 'In-worker bf16 enabled',
        fill: '#f0fdf4',
        stroke: '#16a34a',
        titleColor: '#166534',
        slotColor: '#15803d',
        dashed: false,
      },
      {
        name: 'AMD Turin (Blocked)',
        machine: 'c4d-standard-8',
        countLabel: '2 VMs blocked',
        vcpuLabel: '+16 vCPUs denied',
        slotLabel: 'Logged quota errors',
        note: 'Google kept C4D at 72',
        fill: '#fef2f2',
        stroke: '#dc2626',
        titleColor: '#991b1b',
        slotColor: '#b91c1c',
        dashed: true,
      },
      {
        name: 'Cluster Totals',
        machine: 'Active during test',
        countLabel: '10 active VMs',
        vcpuLabel: '88 / 100 active',
        slotLabel: '84 total (72 + 12 GPU)',
        note: '104 requested if raised',
        fill: '#fffbeb',
        stroke: '#d97706',
        titleColor: '#92400e',
        slotColor: '#92400e',
        dashed: false,
      },
    ],
  },
  {
    id: 'final',
    shortLabel: 'Final 100 vCPU (aaa462f + b90e04e)',
    commit: 'aaa462f + b90e04e · cluster_v5.yaml',
    title: '100-vCPU Full Cluster (v5 & v6)',
    summary: 'Google kept C4D at 72 and total CPUs at 100; the last 12 vCPUs came from Intel C4 (AMX bf16).',
    statusClass: 'is-ok',
    badgeClass: 'badge-green',
    activeVcpus: 100,
    requestedVcpus: 100,
    c4dActive: 72,
    c4Vcpus: 12,
    blockedVcpus: 0,
    cpuSlots: 84,
    gpuWorkers: 13,
    cards: [
      {
        name: 'GPU Head',
        machine: 'g2-standard-16 + 1 L4',
        countLabel: '1 VM',
        vcpuLabel: '16 vCPUs',
        slotLabel: '13 GPU workers (b90e04e)',
        note: '12 initially (aaa462f)',
        fill: '#eff6ff',
        stroke: '#2563eb',
        titleColor: '#1e3a8a',
        slotColor: '#1e40af',
        dashed: false,
      },
      {
        name: 'AMD Turin Workers',
        machine: 'c4d-standard-8',
        countLabel: '9 VMs',
        vcpuLabel: '72 vCPUs (9 × 8)',
        slotLabel: '72 CPU self-play slots',
        note: 'bf16 on CPU',
        fill: '#f0fdf4',
        stroke: '#16a34a',
        titleColor: '#166534',
        slotColor: '#15803d',
        dashed: false,
      },
      {
        name: 'Intel C4 Workers',
        machine: '1 c4-std-8 + 1 c4-std-4',
        countLabel: '2 VMs (8 + 4)',
        vcpuLabel: '12 vCPUs',
        slotLabel: '12 CPU self-play slots',
        note: 'AMX bf16; fills to 100',
        fill: '#faf5ff',
        stroke: '#7c3aed',
        titleColor: '#5b21b6',
        slotColor: '#6d28d9',
        dashed: false,
      },
      {
        name: 'Full Quota Used',
        machine: '12 VMs total (1 + 11)',
        countLabel: '100 / 100 vCPUs',
        vcpuLabel: '84 CPU slots + 13 GPU',
        slotLabel: '97 self-play workers',
        note: '96 before b90e04e',
        fill: '#f0fdf4',
        stroke: '#16a34a',
        titleColor: '#166534',
        slotColor: '#15803d',
        dashed: false,
      },
    ],
  },
];

const stageId = ref('final');
const activeStage = computed(() => STAGES.find((s) => s.id === stageId.value) ?? STAGES[3]);

// Map 0..108 vCPUs across 740 px width
function vcpuW(vcpus) {
  return (vcpus / 108) * 740;
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

.controls-group {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}

.stage-buttons {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
}

.stage-btn {
  font-family: inherit;
  font-size: 12.5px;
  font-weight: 600;
  padding: 6px 11px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.stage-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.stage-btn.active {
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

.status-bar.is-ok {
  background: #f0fdf4;
  border-color: #86efac;
}

.status-bar.is-warn {
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

.badge-blue {
  background: #dbeafe;
  color: #1e40af;
  border: 1px solid #93c5fd;
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
