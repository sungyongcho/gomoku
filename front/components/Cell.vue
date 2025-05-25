<script setup lang="ts">
import type { Stone } from "~/types/game";

const props = defineProps({
  x: {
    type: Number,
    default: 0,
  },
  y: {
    type: Number,
    default: 0,
  },
  stone: {
    type: String as PropType<Stone>,
    default: "",
  },
});
const emit = defineEmits(["put", "evaluate"]);
const onClickCell = () => {
  if (isAiThinking.value) return;
  emit("put", { x: props.x, y: props.y });
};
const onEvaluate = () => {
  // get evaluation
  emit("evaluate", { x: props.x, y: props.y });
};
const onMouseLeave = () => {
  // clear previous evaluation
  emit("evaluate", undefined);
};
const { turn, gameOver, histories, isAiThinking, _histories } =
  storeToRefs(useGameStore());
const lastHistory = computed(() => _histories.value.at(-1));
</script>

<template>
  <button
    class="relative flex aspect-square h-[calc(min(calc(100vw),calc(100vh-64px-105px-42px-20px-80px))/19)] items-center justify-center -sm:h-[calc(min(calc(100vw-25px),calc(100vh-118px-105px-42px-20px-30px))/19)] [&_.previewStone]:hover:block"
    :class="{
      ['cursor-wait']: isAiThinking,
    }"
    @click="onClickCell"
    @contextmenu.prevent="onEvaluate"
    @mouseleave="onMouseLeave"
    :disabled="stone !== '.' || gameOver || isAiThinking"
  >
    <hr
      v-if="x < 18"
      class="absolute right-0 w-1/2 border-[1px] border-black"
    />
    <hr v-if="x > 0" class="absolute left-0 w-1/2 border-[1px] border-black" />
    <hr v-if="y > 0" class="absolute top-0 h-1/2 border-[1px] border-black" />
    <hr
      v-if="y < 18"
      class="absolute bottom-0 h-1/2 border-[1px] border-black"
    />
    <small
      v-if="x == 0"
      class="absolute -left-[10px] w-1/2 text-center text-[calc(min(100vw,calc(100vh-64px-105px-42px-20px-80px))/19*0.35)] font-bold -sm:text-[10px]"
    >
      {{ y + 1 }}
    </small>
    <small
      v-if="y == 0"
      class="absolute -top-[10px] h-1/2 text-center text-[calc(min(100vw,calc(100vh-64px-105px-42px-20px-80px))/19*0.35)] font-bold -sm:text-[10px]"
    >
      {{ String.fromCharCode("A".charCodeAt(0) + x) }}
    </small>
    <span
      v-if="stone !== '.'"
      class="absolute z-10 box-content aspect-square h-[calc(min(100vw,calc(100vh-64px-105px-42px-20px-80px))/19*0.85)] rounded-[50%] bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] shadow-[0_2px_6px_1px_#78716c] -sm:h-[calc(min(calc(100vw-25px),calc(100vh-118px-105px-42px-20px-30px))/19*0.85)]"
      :class="{
        'from-white via-white to-gray-300': stone == 'O',
        'from-gray-600 via-gray-900 to-black': stone == 'X',
        ['z-[11] border-[4px] border-red-500']:
          lastHistory?.coordinate.x === x && lastHistory?.coordinate.y === y,
      }"
    ></span>
    <!-- preview stone -->
    <span
      v-else-if="!gameOver && !isAiThinking"
      class="previewStone absolute z-10 hidden aspect-square h-[calc(min(100vw,calc(100vh-64px-105px-42px-20px-80px))/19*0.85)] rounded-[50%] opacity-50 -sm:h-[calc(min(calc(100vw-25px),calc(100vh-118px-105px-42px-20px-30px))/19*0.85)]"
      :class="{ 'bg-white': turn == 'O', 'bg-black': turn == 'X' }"
    ></span>
  </button>
</template>
