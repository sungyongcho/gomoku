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
const { turn, gameOver, isShowEval } = storeToRefs(useGameStore());
</script>

<template>
  <button
    class="relative flex h-[calc(min(78vw,78vh)/19)] w-[calc(min(78vw,78vh)/19)] items-center justify-center -lg:h-[calc(min(90vw-130px,90vh-130px)/19)] -lg:w-[calc(min(90vw-84px,90vh-84px)/19)] -sm:h-[calc(min(94vw,94vh)/19)] -sm:w-[calc(min(94vw,94vh)/19)] [&_.previewStone]:hover:block"
    @click="onClickCell"
    @contextmenu.prevent="onEvaluate"
    @mouseleave="onMouseLeave"
    :disabled="stone !== '.' || gameOver ? true : false"
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
      class="absolute -left-1/2 w-1/2 text-center text-[13px] font-bold"
    >
      {{ y + 1 }}
    </small>
    <small
      v-if="y == 0"
      class="absolute -top-1/2 h-1/2 text-center text-[13px] font-bold"
    >
      {{ String.fromCharCode("A".charCodeAt(0) + x) }}
    </small>
    <span
      v-if="stone !== '.'"
      class="absolute z-10 h-[calc(min(70vw,70vh)/19)] w-[calc(min(70vw,70vh)/19)] rounded-[50%] shadow-[0_2px_6px_1px_#78716c] -lg:h-[calc(min(70vw,70vh)/19)] -lg:w-[calc(min(70vw,70vh)/19)] -sm:h-[calc(min(80vw,80vh)/19)] -sm:w-[calc(min(80vw,80vh)/19)]"
      :class="{ 'bg-white': stone == 'O', 'bg-black': stone == 'X' }"
    ></span>
    <span
      v-else-if="!gameOver"
      class="previewStone absolute z-10 hidden h-[calc(min(70vw,70vh)/19)] w-[calc(min(70vw,70vh)/19)] rounded-[50%] opacity-50 -lg:h-[calc(min(70vw,70vh)/19)] -lg:w-[calc(min(70vw,70vh)/19)] -sm:h-[calc(min(80vw,80vh)/19)] -sm:w-[calc(min(80vw,80vh)/19)]"
      :class="{ 'bg-white': turn == 'O', 'bg-black': turn == 'X' }"
    ></span>
  </button>
</template>
