<template>
  <div class="h-screen">
    <MHeader non-game-page class="fixed left-0 top-0 z-10 w-full" />
    <slot></slot>
    <MFooter non-game-page class="fixed bottom-0 left-0 z-10 w-full" />
  </div>
</template>
