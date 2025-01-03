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
});
const emit = defineEmits(["update:modelValue"]);
const { modelValue: _modelValue } = useVModels(props, emit);
const id = useId();
</script>

<template>
  <div class="flex flex-col gap-2">
    <label :for="id"> {{ label }} </label>
    <ButtonGroup>
      <Button
        v-for="option in options"
        :severity="option.value === _modelValue ? 'primary' : 'secondary'"
        :key="option.value"
        :label="option.label"
        @click="_modelValue = option.value"
      />
    </ButtonGroup>
  </div>
</template>
