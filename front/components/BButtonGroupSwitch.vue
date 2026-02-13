<script setup lang="ts">
import { useVModels } from "@vueuse/core";

const props = defineProps({
  modelValue: {
    type: [Number, String],
    default: false,
  },
  options: {
    type: Array as PropType<{ value: string | number; label: string }[]>,
    default: [],
  },
  label: {
    type: String,
    default: "",
  },
  name: {
    type: String,
    default: "",
  },
  disabled: {
    type: Boolean,
    default: false,
  },
});
const emit = defineEmits(["update:modelValue"]);
const { modelValue: _modelValue } = useVModels(props, emit);
const id = useId();
</script>

<template>
  <div
    class="flex flex-col gap-2 transition-opacity"
    :class="{
      'opacity-60 cursor-not-allowed pointer-events-none select-none': disabled,
    }"
  >
    <label :for="id" :class="{ 'text-gray-500': disabled }"> {{ label }} </label>
    <ButtonGroup>
      <Button
        v-for="option in options"
        :severity="option.value === _modelValue ? 'primary' : 'secondary'"
        :key="option.value"
        :label="option.label"
        :disabled="disabled"
        @click="disabled ? undefined : (_modelValue = option.value)"
      />
    </ButtonGroup>
  </div>
</template>
