<template>
  <div class="standalone-diagram-page">
    <component :is="activeComponent" v-if="activeComponent" />
    <div v-else class="error-msg">
      Diagram not found: {{ diagramId }}
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue';
import { useRoute } from 'vue-router';
// Import diagram components explicitly (or let Nuxt auto-import handle it if configured globally, but explicit is safer here for dynamic resolution)
import Diagram01 from '~/components/diagrams/Diagram01.vue';
import Diagram02 from '~/components/diagrams/Diagram02.vue';
import Diagram03 from '~/components/diagrams/Diagram03.vue';
import Diagram04 from '~/components/diagrams/Diagram04.vue';
import Diagram05 from '~/components/diagrams/Diagram05.vue';
import Diagram06 from '~/components/diagrams/Diagram06.vue';
import Diagram07 from '~/components/diagrams/Diagram07.vue';
import Diagram08 from '~/components/diagrams/Diagram08.vue';
import Diagram09 from '~/components/diagrams/Diagram09.vue';
import Diagram10 from '~/components/diagrams/Diagram10.vue';
import Diagram11 from '~/components/diagrams/Diagram11.vue';
import Diagram12 from '~/components/diagrams/Diagram12.vue';
import Diagram13 from '~/components/diagrams/Diagram13.vue';
import Diagram14 from '~/components/diagrams/Diagram14.vue';
import Diagram15 from '~/components/diagrams/Diagram15.vue';
import Diagram16 from '~/components/diagrams/Diagram16.vue';
import Diagram17 from '~/components/diagrams/Diagram17.vue';
import Diagram18 from '~/components/diagrams/Diagram18.vue';
import Diagram19 from '~/components/diagrams/Diagram19.vue';
import Diagram20 from '~/components/diagrams/Diagram20.vue';
import Diagram21 from '~/components/diagrams/Diagram21.vue';
import Diagram22 from '~/components/diagrams/Diagram22.vue';
import Diagram23 from '~/components/diagrams/Diagram23.vue';
import Diagram24 from '~/components/diagrams/Diagram24.vue';
import Diagram25 from '~/components/diagrams/Diagram25.vue';
import Diagram26 from '~/components/diagrams/Diagram26.vue';
import Diagram27 from '~/components/diagrams/Diagram27.vue';
import Diagram28 from '~/components/diagrams/Diagram28.vue';
import Diagram29 from '~/components/diagrams/Diagram29.vue';
import Diagram30 from '~/components/diagrams/Diagram30.vue';
import Diagram31 from '~/components/diagrams/Diagram31.vue';
import Diagram32 from '~/components/diagrams/Diagram32.vue';
import Diagram33 from '~/components/diagrams/Diagram33.vue';
import Diagram34 from '~/components/diagrams/Diagram34.vue';
import Diagram35 from '~/components/diagrams/Diagram35.vue';
import Diagram36 from '~/components/diagrams/Diagram36.vue';
import Diagram37 from '~/components/diagrams/Diagram37.vue';
import Diagram38 from '~/components/diagrams/Diagram38.vue';
import Diagram39 from '~/components/diagrams/Diagram39.vue';
import Diagram40 from '~/components/diagrams/Diagram40.vue';
import Diagram41 from '~/components/diagrams/Diagram41.vue';
import Diagram42 from '~/components/diagrams/Diagram42.vue';
import Diagram43 from '~/components/diagrams/Diagram43.vue';

const route = useRoute();
const diagramId = computed(() => String(route.params.id ?? ''));

const componentMap = {
  'diagram_01': Diagram01,
  'diagram_02': Diagram02,
  'diagram_03': Diagram03,
  'diagram_04': Diagram04,
  'diagram_05': Diagram05,
  'diagram_06': Diagram06,
  'diagram_07': Diagram07,
  'diagram_08': Diagram08,
  'diagram_09': Diagram09,
  'diagram_10': Diagram10,
  'diagram_11': Diagram11,
  'diagram_12': Diagram12,
  'diagram_13': Diagram13,
  'diagram_14': Diagram14,
  'diagram_15': Diagram15,
  'diagram_16': Diagram16,
  'diagram_17': Diagram17,
  'diagram_18': Diagram18,
  'diagram_19': Diagram19,
  'diagram_20': Diagram20,
  'diagram_21': Diagram21,
  'diagram_22': Diagram22,
  'diagram_23': Diagram23,
  'diagram_24': Diagram24,
  'diagram_25': Diagram25,
  'diagram_26': Diagram26,
  'diagram_27': Diagram27,
  'diagram_28': Diagram28,
  'diagram_29': Diagram29,
  'diagram_30': Diagram30,
  'diagram_31': Diagram31,
  'diagram_32': Diagram32,
  'diagram_33': Diagram33,
  'diagram_34': Diagram34,
  'diagram_35': Diagram35,
  'diagram_36': Diagram36,
  'diagram_37': Diagram37,
  'diagram_38': Diagram38,
  'diagram_39': Diagram39,
  'diagram_40': Diagram40,
  'diagram_41': Diagram41,
  'diagram_42': Diagram42,
  'diagram_43': Diagram43,
};

const activeComponent = computed(() => componentMap[diagramId.value]);

// Setup minimal metadata for standalone view
definePageMeta({
  layout: false
});

useHead(() => ({
  title: `Diagram ${diagramId.value}`,
  bodyAttrs: {
    class: 'overflow-hidden m-0 p-0 bg-[#f8fafc]'
  }
}));
</script>

<style scoped>
.standalone-diagram-page {
  min-height: 100vh;
  width: 100vw;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f8fafc;
  margin: 0;
  padding: 0;
}

.standalone-diagram-page :deep(.diagram-wrapper) {
  width: 100%;
  max-width: 100%;
  height: 100vh;
  padding: 2vw;
  box-sizing: border-box;
}

.standalone-diagram-page :deep(svg) {
  max-height: 80vh;
  width: auto;
  margin: 0 auto;
}

.error-msg {
  color: #dc2626;
  font-weight: 600;
  font-size: 1.2rem;
}
</style>
