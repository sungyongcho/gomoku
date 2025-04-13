<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from "vue";
import { pipe, range, map, toArray } from "@fxts/core";

const props = defineProps({
  testData: {
    type: Object as PropType<TestCase>,
    default: pipe(
      range(19),
      map((_) =>
        pipe(
          range(19),
          map((_) => ({ stone: "." as Stone })),
          toArray,
        ),
      ),
      toArray,
    ),
  },
  load: {
    type: Boolean,
    default: false,
  },
});

const canvasRef = ref<HTMLCanvasElement | null>(null);

const drawBoard = () => {
  const canvas = canvasRef.value;
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const size = 19;
  const width = canvas.width;
  const height = canvas.height;

  // 여백 설정 (돌이 캔버스를 벗어나지 않게)
  const padding = width * 0.03;
  const boardSize = width - padding * 2;
  const cellSize = boardSize / (size - 1);

  ctx.clearRect(0, 0, width, height);

  // 바둑판 배경
  ctx.fillStyle = "#DEB887"; // 약간 노란 나무색
  ctx.fillRect(0, 0, width, height);

  // 선 그리기
  ctx.strokeStyle = "#000";
  ctx.lineWidth = 1;

  for (let i = 0; i < size; i++) {
    const pos = padding + i * cellSize;

    // 수평선
    ctx.beginPath();
    ctx.moveTo(padding, pos);
    ctx.lineTo(width - padding, pos);
    ctx.stroke();

    // 수직선
    ctx.beginPath();
    ctx.moveTo(pos, padding);
    ctx.lineTo(pos, height - padding);
    ctx.stroke();
  }

  // 돌 그리기
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const stone = props.testData.boardData[row][col].stone;

      if (stone !== "O" && stone !== "X") continue; // "."인 경우 스킵

      const x = padding + col * cellSize;
      const y = padding + row * cellSize;
      const radius = cellSize * 0.4;

      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = stone === "O" ? "#fff" : "#000";
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }
  }
};

const resizeCanvas = () => {
  const canvas = canvasRef.value;
  if (!canvas || !canvas.parentElement) return;

  const parent = canvas.parentElement;
  const size = parent.clientWidth;

  canvas.width = size;
  canvas.height = size;

  drawBoard();
};

onMounted(() => {
  window.addEventListener("resize", resizeCanvas);
});

onBeforeUnmount(() => {
  window.removeEventListener("resize", resizeCanvas);
});

watch(
  () => [props.testData, props.load],
  ([_, isLoad]) => {
    resizeCanvas();
  },
);
</script>

<template>
  <div class="go-board-wrapper">
    <canvas ref="canvasRef" class="go-board-canvas" />
  </div>
</template>

<style scoped>
.go-board-wrapper {
  width: 100%;
  aspect-ratio: 1 / 1;
}
.go-board-canvas {
  width: 100%;
  height: 100%;
  display: block;
}
</style>
