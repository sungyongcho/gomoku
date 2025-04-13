<script setup lang="ts">
import { useWebSocket, type WebSocketStatus } from "@vueuse/core";

import type {
  RequestType,
  SocketMoveResponse,
  SocketMoveRequest,
} from "~/types/game";

definePageMeta({
  layout: "game",
});

const { isAiThinking } = storeToRefs(useGameStore());

const {
  deleteLastHistory,
  initGame,
  addStoneToBoardData,
  initialBoard,
  getPlayerTotalCaptured,
} = useGameStore();

const lastHistory = computed(() => histories.value.at(-1));
const { doAlert } = useAlertStore();

const { data, send, close, status } = useWebSocket(
  `ws://${window.location.hostname}:8005/ws/debug`,
  {
    autoReconnect: {
      retries: 3,
      delay: 500,
      onFailed() {
        doAlert(
          "Error",
          "WebSocket connection failed. Please refresh the page to retry",
          "Warn",
        );
        isAiThinking.value = false;
      },
    },
  },
);

const onSendData = (type: RequestType, testCase: TestCase) => {
  isAiThinking.value = true;
  const lastPlay = testCase.histories.at(-1);
  send(
    JSON.stringify({
      type,
      difficulty: "hard",
      nextPlayer: lastPlay ? (lastPlay?.stone === "X" ? "O" : "X") : "O",
      goal: 5,
      lastPlay: lastPlay
        ? {
            coordinate: {
              x: lastPlay.coordinate.x,
              y: lastPlay.coordinate.y,
            },
            stone: lastPlay.stone,
          }
        : undefined,
      board: testCase.boardData.map((row) => row.map((col) => col.stone)),
      scores: [
        { player: "X", score: getPlayerTotalCaptured(testCase.histories, "X") },
        { player: "O", score: getPlayerTotalCaptured(testCase.histories, "O") },
      ],
    } as SocketMoveRequest),
  );
};

const activeIndex = ref();
const testCases = ref<TestCase>([]);

const loadTestCases = () => {
  const files = import.meta.glob(
    "@/assets/testCases/**/+(init|expected).json",
    {
      eager: true,
      import: "default", // JSON은 default export
    },
  );

  const result: Record<
    string,
    {
      init?: TestCase;
      expected?: TestCase;
    }
  > = {};

  for (const path in files) {
    const match = path.match(/testCases\/([^/]+)\/(init|expected)\.json$/);
    if (!match) continue;

    const [, testName, type] = match;
    if (!result[testName]) {
      result[testName] = {};
    }
    result[testName][type] = files[path];
  }

  Object.keys(result).forEach((testName) => {
    result[testName]["evaluated"] = {
      ...result[testName]["init"],
      boardData: initialBoard(),
      histories: [],
      turn: "X",
    };
  });

  return result;
};

onMounted(() => {
  testCases.value = loadTestCases();
});

onUnmounted(() => {
  close();
});

const triggeredTestLabel = ref("");
const onTest = (label: string, idx: number) => {
  const initState = testCases.value[label]["init"];
  const histories = initState.histories;
  triggeredTestLabel.value = label;
  onSendData("move", initState);
};

watch(data, (rawData) => {
  if (!data.value) return;
  if (!triggeredTestLabel.value) return;

  try {
    const res: SocketMoveResponse =
      typeof rawData === "string" ? JSON.parse(rawData) : rawData;

    if (res.type === "error") {
      console.error(res);
      doAlert("Caution", res.error, "Warn");
      return;
    }

    testCases.value[triggeredTestLabel.value]["evaluated"] = {
      ...testCases.value[triggeredTestLabel.value]["init"],
      boardData: res.board.map((row) => row.map((col) => ({ stone: col }))),
    };

    triggeredTestLabel.value = "";
  } catch (error) {
    console.error("Error processing WebSocket data:", error);
    doAlert(
      "Error",
      "An unexpected error occurred while processing data.",
      "Warn",
    );
  }
});
</script>
<template>
  <main class="relative">
    <section class="mx-auto mt-10 max-w-[1200px] px-6 -sm:my-[80px]">
      <h1 class="flex items-center gap-4 text-4xl">
        <Button size="small" class="leading-none">
          Trigger All Tests <i class="pi pi-bolt"></i>
        </Button>

        Evaluation test cases
      </h1>

      <div class="card">
        <Accordion v-model="activeIndex">
          <AccordionPanel
            v-for="([label, testCaseData], testIndex) in Object.entries(
              testCases,
            )"
            @click="activeIndex = testIndex"
            :key="testIndex"
            :value="testIndex"
          >
            <AccordionHeader>
              <div class="flex items-center gap-8">
                <Badge
                  value="Passed"
                  severity="success"
                  v-if="
                    JSON.stringify(testCaseData.evaluated.boardData) ===
                    JSON.stringify(testCaseData.expected.boardData)
                  "
                />
                <Badge value="Not Passed" severity="danger" v-else />

                <Button
                  size="small"
                  severity="secondary"
                  icon="pi pi-bolt"
                  @click.stop="onTest(label, testIndex)"
                  :loading="triggeredTestLabel === label"
                  :disabled="triggeredTestLabel === label"
                  label="Test"
                />
                <span>
                  {{ label }}
                </span>
              </div>
            </AccordionHeader>
            <AccordionContent>
              <div class="flex items-center justify-between">
                <figure class="w-[30%]">
                  <p class="text-center">Initial Board</p>
                  <TestBoard
                    :testData="testCaseData.init"
                    :load="testIndex === activeIndex"
                  />
                </figure>

                <figure class="w-[30%]">
                  <p class="text-center">Evaluated Board</p>
                  <TestBoard
                    :testData="testCaseData.evaluated"
                    :load="testIndex === activeIndex"
                  />
                </figure>

                <figure class="w-[30%]">
                  <p class="text-center">Expected Board</p>
                  <TestBoard
                    :testData="testCaseData.expected"
                    :load="testIndex === activeIndex"
                  />
                </figure>
              </div>
            </AccordionContent>
          </AccordionPanel>
        </Accordion>
      </div>
    </section>
  </main>
</template>
