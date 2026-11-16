<template>
  <div class="diagram-wrapper">
    <div class="container">
      <h1>IEEE 754 Float32 Layout: Normal vs. Subnormal (Denormal)</h1>

      <div class="controls-panel">
        <div class="preset-row">
          <button
            v-for="preset in presets"
            :key="preset.label"
            type="button"
            class="preset-btn"
            :class="{ active: rawBits === preset.bits }"
            @click="setBits(preset.bits)"
          >
            {{ preset.label }}
          </button>
          <button
            type="button"
            class="flush-btn"
            :class="{ on: flushDenormals }"
            @click="flushDenormals = !flushDenormals"
          >
            set_flush_denormal({{ flushDenormals ? 'True' : 'False' }})
          </button>
        </div>

        <div class="sliders-row">
          <label class="slider-item">
            <span>Scale <code>2^k</code>: <strong>{{ scaleLabel }}</strong></span>
            <input
              :value="scaleExp"
              type="range"
              min="-150"
              max="0"
              step="1"
              class="range-input"
              @input="onScaleSlide(Number($event.target.value))"
            />
          </label>
          <button type="button" class="sign-btn" @click="toggleBit(31)">
            Sign: {{ signBit === 0 ? '+ (0)' : '− (1)' }}
          </button>
          <span class="hint-text">Click any bit below to toggle</span>
        </div>
      </div>

      <svg
        class="diagram-svg"
        viewBox="0 0 780 398"
        xmlns="http://www.w3.org/2000/svg"
        role="img"
        aria-label="Interactive IEEE 754 float32 32-bit layout showing normal, subnormal, and zero ranges"
      >
        <!-- Header note -->
        <rect x="20" y="8" width="740" height="24" rx="6" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2" />
        <text x="390" y="24" text-anchor="middle" font-size="11.5" font-weight="600" fill="#334155">
          Exact IEEE 754 binary32 bit layout · Measured v5 Intel C4 timings from v5-small-6x96-cont9.yaml
        </text>

        <!-- Group headers -->
        <text x="34" y="50" text-anchor="middle" font-size="11" font-weight="700" fill="#7c3aed">Sign (1)</text>
        <text x="146" y="50" text-anchor="middle" font-size="11" font-weight="700" fill="#2563eb">
          Exponent E (8 bits, bias 127)
        </text>
        <text x="498" y="50" text-anchor="middle" font-size="11" font-weight="700" fill="#059669">
          Fraction f (23 bits)
        </text>

        <!-- 32 clickable bit boxes -->
        <g
          v-for="bit in bitCells"
          :key="bit.pos"
          class="bit-cell"
          @click="toggleBit(bit.pos)"
        >
          <rect
            :x="bit.x"
            y="58"
            width="20"
            height="32"
            rx="4"
            :fill="bit.val ? bit.activeFill : '#ffffff'"
            :stroke="bit.stroke"
            stroke-width="1.5"
          />
          <text
            :x="bit.x + 10"
            y="79"
            text-anchor="middle"
            font-size="13"
            font-weight="700"
            :fill="bit.val ? bit.activeText : '#64748b'"
          >
            {{ bit.val }}
          </text>
          <text
            v-if="bit.showIndex"
            :x="bit.x + 10"
            y="103"
            text-anchor="middle"
            font-size="9.5"
            fill="#64748b"
          >
            {{ bit.pos }}
          </text>
        </g>

        <!-- Field value summary strip -->
        <text x="34" y="121" text-anchor="middle" font-size="10.5" font-weight="600" fill="#5b21b6">
          s = {{ signBit }}
        </text>
        <text x="146" y="121" text-anchor="middle" font-size="10.5" font-weight="600" fill="#1e40af">
          E = {{ expVal }} ({{ expHex }})
        </text>
        <text x="498" y="121" text-anchor="middle" font-size="10.5" font-weight="600" fill="#065f46">
          f = {{ fracVal }} / 2²³ ({{ fracHex }}) · Hex: {{ fullHex }}
        </text>

        <!-- Decoded formula & status card -->
        <rect
          x="20"
          y="134"
          width="740"
          height="108"
          rx="8"
          :fill="statusStyle.bg"
          :stroke="statusStyle.border"
          stroke-width="1.5"
        />

        <rect x="34" y="146" width="210" height="24" rx="5" :fill="statusStyle.badgeBg" />
        <text x="139" y="162" text-anchor="middle" font-size="11.5" font-weight="700" fill="#ffffff">
          {{ statusStyle.badgeText }}
        </text>

        <text x="260" y="162" font-size="12" font-weight="700" :fill="statusStyle.titleColor">
          {{ implicitNote }}
        </text>

        <text x="34" y="190" font-size="12.5" font-weight="600" fill="#0f172a">
          Formula: {{ decodedFormula }}
        </text>
        <text x="34" y="212" font-size="13" font-weight="700" fill="#0f172a">
          Encoded float32: {{ formattedValue }}
        </text>
        <text
          v-if="flushDenormals && classification === 'subnormal'"
          x="390"
          y="212"
          font-size="12.5"
          font-weight="700"
          fill="#047857"
        >
          → CPU sees (FTZ/DAZ): {{ signBit ? '-0.0' : '+0.0' }}
        </text>
        <text x="34" y="231" font-size="11" :fill="statusStyle.noteColor">
          {{ hardwareNote }}
        </text>

        <!-- Logarithmic scale bar -->
        <rect x="20" y="254" width="740" height="132" rx="8" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.4" />
        <text x="34" y="274" font-size="12.5" font-weight="700" fill="#0f172a">
          Float32 Magnitude Axis (log₂ scale near the subnormal boundary)
        </text>

        <!-- Zero box -->
        <rect x="44" y="294" width="54" height="24" rx="4" fill="#f1f5f9" stroke="#64748b" stroke-width="1.3" />
        <text x="71" y="310" text-anchor="middle" font-size="11" font-weight="700" fill="#334155">0.0</text>

        <!-- Subnormal band: -149 to -126 -->
        <rect x="114" y="294" width="230" height="24" rx="4" fill="#fef2f2" stroke="#dc2626" stroke-width="1.4" />
        <text x="229" y="310" text-anchor="middle" font-size="11" font-weight="700" fill="#991b1b">
          Subnormal: 2⁻¹⁴⁹ .. &lt; 2⁻¹²⁶ (E = 0, f ≠ 0)
        </text>

        <!-- Normal band: -126 to 0+ -->
        <rect x="344" y="294" width="388" height="24" rx="4" fill="#eff6ff" stroke="#2563eb" stroke-width="1.4" />
        <text x="538" y="310" text-anchor="middle" font-size="11" font-weight="700" fill="#1e40af">
          Normal: 2⁻¹²⁶ .. 2¹²⁷·(2−2⁻²³) (1 ≤ E ≤ 254)
        </text>

        <!-- Boundary ticks -->
        <line x1="114" y1="290" x2="114" y2="324" stroke="#991b1b" stroke-width="1.4" />
        <text x="114" y="337" text-anchor="middle" font-size="10" fill="#991b1b">2⁻¹⁴⁹ (1.40e-45)</text>

        <line x1="344" y1="286" x2="344" y2="326" stroke="#0f172a" stroke-width="1.8" stroke-dasharray="3 2" />
        <text x="344" y="337" text-anchor="middle" font-size="10" font-weight="700" fill="#0f172a">
          2⁻¹²⁶ (1.18e-38)
        </text>

        <line x1="654" y1="290" x2="654" y2="324" stroke="#1e40af" stroke-width="1.4" />
        <text x="654" y="337" text-anchor="middle" font-size="10" fill="#1e40af">2⁰ = 1.0</text>

        <!-- Current value marker on axis -->
        <circle
          :cx="markerX"
          cy="288"
          r="6"
          :fill="statusStyle.markerColor"
          stroke="#ffffff"
          stroke-width="1.8"
        />

        <!-- Measured v5 footer inside card -->
        <rect x="34" y="348" width="712" height="28" rx="5" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1" />
        <text x="390" y="366" text-anchor="middle" font-size="11" fill="#334155">
          v5 (~281,000 denormal weights): Intel C4 float32 206–258 ms/pos vs. 7 ms AMD C4D · 10 ms on Intel with flush enabled
        </text>
      </svg>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const presets = [
  { label: '1.0', bits: 0x3f800000 },
  { label: '2⁻¹²⁶ (min normal)', bits: 0x00800000 },
  { label: '2⁻¹²⁷ (subnormal)', bits: 0x00400000 },
  { label: '2⁻¹⁴⁹ (min subnormal)', bits: 0x00000001 },
  { label: '0.0', bits: 0x00000000 },
];

const rawBits = ref(0x00400000); // Start on 2^-127 so subnormal layout is immediately visible
const flushDenormals = ref(false);

const signBit = computed(() => (rawBits.value >>> 31) & 1);
const expVal = computed(() => (rawBits.value >>> 23) & 0xff);
const fracVal = computed(() => rawBits.value & 0x7fffff);

const expHex = computed(() => `0x${expVal.value.toString(16).padStart(2, '0')}`);
const fracHex = computed(() => `0x${fracVal.value.toString(16).padStart(6, '0')}`);
const fullHex = computed(() => `0x${(rawBits.value >>> 0).toString(16).padStart(8, '0')}`);

const floatValue = computed(() => {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint32(0, rawBits.value >>> 0, false);
  return view.getFloat32(0, false);
});

const classification = computed(() => {
  if (expVal.value === 255) {
    return fracVal.value === 0 ? 'inf' : 'nan';
  }
  if (expVal.value === 0) {
    return fracVal.value === 0 ? 'zero' : 'subnormal';
  }
  return 'normal';
});

const scaleExp = computed(() => {
  if (classification.value === 'zero') return -150;
  if (classification.value === 'inf' || classification.value === 'nan') return 0;
  const av = Math.abs(floatValue.value);
  const log2 = Math.floor(Math.log2(av));
  return Math.max(-149, Math.min(0, log2));
});

const scaleLabel = computed(() => {
  if (classification.value === 'zero') return '0';
  if (classification.value === 'inf') return '±∞';
  if (classification.value === 'nan') return 'NaN';
  const k = Math.floor(Math.log2(Math.abs(floatValue.value)));
  return `2^(${k})`;
});

function setBits(b) {
  rawBits.value = b >>> 0;
}

function toggleBit(pos) {
  rawBits.value = (rawBits.value ^ (1 << pos)) >>> 0;
}

function onScaleSlide(k) {
  const s = signBit.value ? 0x80000000 : 0;
  if (k <= -150) {
    rawBits.value = s >>> 0;
  } else if (k <= -127) {
    const shift = k + 149; // 0..22
    rawBits.value = (s | (1 << shift)) >>> 0;
  } else {
    const e = k + 127; // 1..127
    rawBits.value = (s | (e << 23)) >>> 0;
  }
}

const bitCells = computed(() => {
  const cells = [];
  for (let pos = 31; pos >= 0; pos--) {
    const idxFromLeft = 31 - pos;
    let x = 24 + idxFromLeft * 22;
    if (pos <= 30) x += 12;
    if (pos <= 22) x += 12;

    let stroke = '#059669';
    let activeFill = '#d1fae5';
    let activeText = '#065f46';
    if (pos === 31) {
      stroke = '#7c3aed';
      activeFill = '#ede9fe';
      activeText = '#5b21b6';
    } else if (pos >= 23) {
      stroke = '#2563eb';
      activeFill = '#dbeafe';
      activeText = '#1e40af';
    }

    const showIndex = pos === 31 || pos === 30 || pos === 23 || pos === 22 || pos === 0;
    cells.push({
      pos,
      x,
      val: (rawBits.value >>> pos) & 1,
      stroke,
      activeFill,
      activeText,
      showIndex,
    });
  }
  return cells;
});

const statusStyle = computed(() => {
  const cls = classification.value;
  if (cls === 'subnormal') {
    if (flushDenormals.value) {
      return {
        bg: '#ecfdf5',
        border: '#10b981',
        badgeBg: '#059669',
        badgeText: 'SUBNORMAL → FLUSHED (0.0)',
        titleColor: '#065f46',
        noteColor: '#047857',
        markerColor: '#059669',
      };
    }
    return {
      bg: '#fef2f2',
      border: '#ef4444',
      badgeBg: '#dc2626',
      badgeText: 'SUBNORMAL (DENORMAL)',
      titleColor: '#991b1b',
      noteColor: '#b91c1c',
      markerColor: '#dc2626',
    };
  }
  if (cls === 'zero') {
    return {
      bg: '#f8fafc',
      border: '#94a3b8',
      badgeBg: '#475569',
      badgeText: 'EXACT ZERO',
      titleColor: '#334155',
      noteColor: '#475569',
      markerColor: '#475569',
    };
  }
  if (cls === 'inf' || cls === 'nan') {
    return {
      bg: '#fffbeb',
      border: '#f59e0b',
      badgeBg: '#d97706',
      badgeText: cls === 'inf' ? 'INFINITY (E = 255)' : 'NaN (E = 255)',
      titleColor: '#92400e',
      noteColor: '#b45309',
      markerColor: '#d97706',
    };
  }
  return {
    bg: '#eff6ff',
    border: '#3b82f6',
    badgeBg: '#2563eb',
    badgeText: 'NORMAL FLOAT32',
    titleColor: '#1e40af',
    noteColor: '#1d4ed8',
    markerColor: '#2563eb',
  };
});

const implicitNote = computed(() => {
  const cls = classification.value;
  if (cls === 'subnormal') return 'E = 0, f ≠ 0 → implicit leading 0 (unbiased exponent fixed at −126)';
  if (cls === 'zero') return 'E = 0, f = 0 → signed zero';
  if (cls === 'inf' || cls === 'nan') return 'E = 255 → special IEEE 754 value (non-finite)';
  return `1 ≤ E ≤ 254 → implicit leading 1 (exponent E − 127 = ${expVal.value - 127})`;
});

const decodedFormula = computed(() => {
  const s = signBit.value === 0 ? '+1' : '−1';
  const cls = classification.value;
  if (cls === 'zero') return `${s} × 0`;
  if (cls === 'inf') return `${s} × ∞`;
  if (cls === 'nan') return 'NaN (E = 255, f ≠ 0)';
  const fracRatio = (fracVal.value / 8388608).toFixed(6);
  if (cls === 'subnormal') {
    return `(${s}) × 2^(−126) × (0 + ${fracRatio})`;
  }
  return `(${s}) × 2^(${expVal.value - 127}) × (1 + ${fracRatio})`;
});

const formattedValue = computed(() => {
  const cls = classification.value;
  if (cls === 'zero') return signBit.value ? '-0.0' : '+0.0';
  if (cls === 'inf') return signBit.value ? '-Infinity' : '+Infinity';
  if (cls === 'nan') return 'NaN';
  const v = floatValue.value;
  if (Math.abs(v) >= 1e-3 && Math.abs(v) < 1e4) {
    return v.toPrecision(7);
  }
  return v.toExponential(6);
});

const hardwareNote = computed(() => {
  const cls = classification.value;
  if (cls === 'subnormal') {
    if (flushDenormals.value) {
      return 'DAZ/FTZ enabled: hardware treats subnormal inputs/outputs as 0.0 without microcode assist (Intel C4: 10 ms/pos).';
    }
    return 'No implicit leading 1: Intel x86 hardware traps to microcode assist (Intel C4: 206–258 ms/pos vs. 10 ms flushed).';
  }
  if (cls === 'zero') {
    return 'Exact zero runs at full hardware pipeline speed with no microcode assist.';
  }
  return 'Normalized significand (1.f) runs in the fast IEEE 754 hardware pipeline.';
});

const markerX = computed(() => {
  const cls = classification.value;
  if (cls === 'zero') return 71;
  if (cls === 'inf' || cls === 'nan') return 732;
  const log2 = Math.log2(Math.abs(floatValue.value));
  if (log2 < -126) {
    // Subnormal zone: x=114 (-149) to x=344 (-126)
    const frac = Math.min(1, Math.max(0, (log2 - -149) / (-126 - -149)));
    return 114 + frac * (344 - 114);
  }
  // Normal zone: x=344 (-126) to x=654 (0) to x=732 (+32+)
  if (log2 <= 0) {
    const frac = (log2 - -126) / 126;
    return 344 + frac * (654 - 344);
  }
  const frac = Math.min(1, log2 / 128);
  return 654 + frac * (732 - 654);
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

.controls-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 10px;
}

.preset-row,
.sliders-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 8px 10px;
}

.preset-btn,
.sign-btn,
.flush-btn {
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  padding: 5px 10px;
  border: 1.5px solid #cbd5e1;
  border-radius: 6px;
  background: #ffffff;
  color: #334155;
  cursor: pointer;
  transition: all 0.15s ease;
}

.preset-btn:hover,
.sign-btn:hover,
.flush-btn:hover {
  background: #f1f5f9;
  border-color: #94a3b8;
}

.preset-btn.active {
  background: #eff6ff;
  border-color: #2563eb;
  color: #1e40af;
}

.flush-btn.on {
  background: #ecfdf5;
  border-color: #10b981;
  color: #065f46;
}

.slider-item {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12.5px;
  color: #334155;
}

.range-input {
  width: 145px;
  max-width: 100%;
  accent-color: #2563eb;
  cursor: pointer;
}

.hint-text {
  font-size: 11.5px;
  color: #64748b;
}

.diagram-svg {
  width: 100%;
  max-width: 100%;
  height: auto;
  display: block;
}

.bit-cell {
  cursor: pointer;
}

.bit-cell:hover rect {
  stroke-width: 2.2;
}

@media (max-width: 480px) {
  h1 {
    font-size: 16.5px;
  }
}
</style>
