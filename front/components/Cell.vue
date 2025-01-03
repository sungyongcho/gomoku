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
  stoneType: {
    type: String as PropType<Stone>,
    default: "",
  },
});
const emit = defineEmits(["put"]);
const onClickCell = () => {
  emit("put", { x: props.x, y: props.y });
};
</script>

<template>
  <button
    class="relative flex h-[calc(min(80vw,80vh)/19)] w-[calc(min(80vw,80vh)/19)] items-center justify-center -lg:h-[calc(min(90vw-84px,90vh-84px)/19)] -lg:w-[calc(min(90vw-84px,90vh-84px)/19)] -sm:h-[calc(min(94vw,94vh)/19)] -sm:w-[calc(min(94vw,94vh)/19)]"
    @click="onClickCell"
    :disabled="stoneType ? true : false"
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
      v-if="stoneType"
      class="absolute z-10 h-[calc(min(68vw,68vh)/19)] w-[calc(min(68vw,68vh)/19)] rounded-[50%] shadow-[0_2px_6px_1px_#78716c] -lg:h-[calc(min(70vw,70vh)/19)] -lg:w-[calc(min(70vw,70vh)/19)] -sm:h-[calc(min(80vw,80vh)/19)] -sm:w-[calc(min(80vw,80vh)/19)]"
      :class="{ 'bg-white': stoneType == 'O', 'bg-black': stoneType == 'X' }"
    ></span>
  </button>
</template>
