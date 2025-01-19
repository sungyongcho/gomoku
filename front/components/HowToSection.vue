<script setup lang="ts"></script>

<template>
  <section
    class="mt-6 rounded-lg border-2 bg-white p-4 text-black [&_b]:font-bold"
  >
    <h2 class="text-2xl font-bold">
      <slot name="title"></slot>
    </h2>

    <h3 class="text-2xl font-bold">
      <slot name="subtitle"></slot>
    </h3>

    <slot name="content"></slot>
  </section>
</template>
