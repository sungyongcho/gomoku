<script setup lang="ts">
import { useVModels } from "@vueuse/core";

const props = defineProps({
  modelValue: {
    type: Boolean,
    default: false,
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
    <ToggleSwitch :id="id" :name="name" v-model="_modelValue" />
  </div>
</template>
