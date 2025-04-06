<script setup lang="ts">
const $router = useRouter();

const onClickBack = () => {
  $router.push("/");
};
</script>

<template>
  <main class="h-full min-h-screen w-full bg-black text-white">
    <div class="mx-auto w-full max-w-[1280px] p-4 pb-0">
      <button @click="onClickBack">
        <i class="pi pi-arrow-left"></i> go back
      </button>
    </div>
    <div
      class="mx-auto grid w-full max-w-[1280px] grid-cols-2 gap-4 p-4 pb-20 pt-0 -sm:grid-cols-1"
    >
      <section
        class="mt-6 rounded-lg border-2 bg-white p-4 text-black [&_b]:font-bold"
      >
        <h2 class="text-2xl font-bold">How to Play</h2>

        <h3 class="mt-4 text-xl font-bold">1. Game Setup:</h3>

        <ul class="ml-4 mt-2 list-disc pl-4">
          <li>The game is played on a grid board (e.g., 15x15 or larger).</li>
          <li>
            One player takes Black stones, and the other takes White stones.
          </li>
        </ul>

        <h3 class="mt-4 text-xl font-bold">2. Turn Order:</h3>
        <ul class="ml-4 mt-2 list-disc pl-4">
          <li>Black always goes first.</li>
          <li>
            Players take turns placing one stone on an empty intersection of the
            board.
          </li>
        </ul>

        <h3 class="mt-4 text-xl font-bold">
          3. Special Rule: Capturing a Pair:
        </h3>
        <ul class="ml-4 mt-2 list-disc pl-4">
          <li>
            A pair capture occurs when two of your opponent's stones are
            directly in a straight line between two of your stones.
          </li>
          <li>
            When this happens, you capture the two opponent stones and remove
            them from the board.
          </li>
          <li>Captures can only be made in pairs (two stones at a time).</li>
        </ul>
      </section>

      <section
        class="mt-6 rounded-lg border-2 bg-white p-4 text-black [&_b]:font-bold"
      >
        <h2 class="text-2xl font-bold">Winning Conditions</h2>

        <h3 class="mt-4 text-xl font-bold">1. Capture Victory:</h3>

        <p class="mt-2 pl-4">
          Achieve the target score by capturing a specified number of pairs
          (e.g., 5 pairs = 10 stones).
        </p>

        <h3 class="mt-4 text-xl font-bold">2. Line Victory:</h3>
        <p class="mt-2 pl-4">
          Arrange five or more stones in an unbroken straight line (horizontal,
          vertical, or diagonal) that cannot be interrupted or captured by the
          opponent.
        </p>
      </section>

      <section
        class="mt-6 rounded-lg border-2 bg-white p-4 text-black [&_b]:font-bold"
      >
        <h2 class="text-2xl font-bold">Placement Restrictions</h2>

        <h3 class="mt-4 text-xl font-bold">
          1. Prohibition of Double-Three (33):
        </h3>

        <div class="mb-10 mt-2 flex gap-4 pl-4">
          <figure class="max-h-[400px] flex-1">
            <img
              src="@/assets/howto/double-three-1.png"
              alt="33 rule"
              class="h-full w-full object-cover"
            />
            <figcaption class="text-center text-gray-800">
              - double three case 1 -
            </figcaption>
          </figure>
          <figure class="max-h-[400px] flex-1">
            <img
              src="@/assets/howto/double-three-2.png"
              alt="33 rule"
              class="h-full w-full object-cover"
            />
            <figcaption class="text-center text-gray-800">
              - double three case 2 -
            </figcaption>
          </figure>
        </div>

        <p class="mt-2 pl-4">
          A player cannot create a double-three (33) when placing a stone. This
          means you cannot make a move that simultaneously creates
          <b>two open lines</b> of three stones (open threes).
        </p>

        <h3 class="mt-4 text-xl font-bold">2. Exception to the 33 Rule:</h3>

        <figure class="mt-2 w-full pl-4">
          <img
            class="h-full w-full object-cover"
            src="@/assets/howto/double-three-exception.png"
            alt="33 rule exception"
          />
          <figcaption class="text-center text-gray-800">
            - double three exception case -
          </figcaption>
        </figure>
        <p class="mt-2 pl-4">
          If the placement of a stone creates a line of five or more stones, the
          33 restriction does not apply.
        </p>
      </section>

      <section
        class="mt-6 rounded-lg border-2 bg-white p-4 text-black [&_b]:font-bold"
      >
        <h2 class="text-2xl font-bold">Additional Notes</h2>

        <p class="mt-2">
          Players must strategically balance offense (capturing pairs or forming
          lines) and defense (blocking the opponent's moves and preventing
          captures). Capturing pairs and avoiding restricted moves (like 33) are
          crucial to mastering the game.
        </p>
      </section>
    </div>
  </main>
</template>
