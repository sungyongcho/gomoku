import type { DocLink, DocFolder, DocItem } from "~/types/docs";

// 아코디언으로 표시할 그룹 목록 (데스크톱/모바일 공유)
export const ACCORDION_GROUPS = new Set([
  "about gomoku",
  "about project",
  "alphazero",
  "minimax",
  "optimization with ai for alphazero",
]);
const GROUP_PRIORITY = [
  "about gomoku",
  "about project",
  "minimax",
  "alphazero",
  "optimization with ai for alphazero",
];

export const useDocsStore = defineStore("docs", () => {
  const docLinks = ref<DocLink[]>([]);
  /** 경로 → 그룹명 (Pinia 직렬화를 위해 plain object 사용) */
  const pathToGroupMap = ref<Record<string, string>>({});
  /** 아코디언에서 열린 그룹 (정규화된 이름). 데스크톱/모바일 공유 */
  const openGroups = ref<string[]>([]);

  const isGroupOpen = (groupName: string): boolean => {
    return openGroups.value.includes(groupName.toLowerCase());
  };

  const toggleGroup = (groupName: string) => {
    const normalized = groupName.toLowerCase();
    const idx = openGroups.value.indexOf(normalized);
    if (idx === -1) {
      openGroups.value = [...openGroups.value, normalized];
    } else {
      openGroups.value = openGroups.value.filter((_, i) => i !== idx);
    }
  };

  const addOpenGroup = (groupName: string) => {
    const normalized = groupName.toLowerCase();
    if (!openGroups.value.includes(normalized)) {
      openGroups.value = [...openGroups.value, normalized];
    }
  };
  
  const fetchDocLinks = async () => {
    const docs = await queryCollection("docs").all();

    // group 필드로 그룹화
    const groupMap = new Map<string, Array<DocItem & { _id?: string }>>();
    const pathMap: Record<string, string> = {};

    for (const doc of docs) {
      const group = doc.group || "Other";

      // 경로 -> 그룹 매핑 저장 (plain object로 Pinia SSR 직렬화 가능)
      pathMap[doc.path] = group;

      if (!groupMap.has(group)) {
        groupMap.set(group, []);
      }

      // doc.id 예: "docs/docs/About Gomoku/00.intro.md"
      const docId = (doc as { id?: string }).id ?? doc.path;

      const docItem: DocItem & { _id?: string } = {
        label: doc.title,
        icon: doc.icon || "pi-file",
        url: doc.path,
        _id: docId,
      };

      groupMap.get(group)!.push(docItem);
    }

    pathToGroupMap.value = pathMap;

    // doc.id에서 파일명 추출 (예: "docs/docs/About Gomoku/00.intro.md" -> "00.intro")
    const getFileNameFromId = (item: DocItem & { _id?: string }): string => {
      const id = item._id || item.url;
      const parts = id.split("/");
      const fileName = parts[parts.length - 1] || id;
      return fileName.replace(/\.md$/, "");
    };
    
    // 자연스러운 정렬을 위한 비교 함수 (숫자 접두사 고려)
    const naturalCompare = (a: string, b: string): number => {
      // 숫자 부분 추출
      const numMatchA = a.match(/^(\d+)/);
      const numMatchB = b.match(/^(\d+)/);
      
      if (numMatchA && numMatchB) {
        // 둘 다 숫자로 시작하는 경우 숫자로 비교
        const numA = parseInt(numMatchA[1], 10);
        const numB = parseInt(numMatchB[1], 10);
        if (numA !== numB) {
          return numA - numB;
        }
        // 숫자가 같으면 나머지 부분 비교
        const restA = a.slice(numMatchA[1].length);
        const restB = b.slice(numMatchB[1].length);
        return restA.localeCompare(restB);
      }
      
      // 숫자로 시작하지 않으면 일반 문자열 비교
      return a.localeCompare(b);
    };
    
    // 파일명 기준 정렬 함수 (id에서 추출한 파일명으로)
    const sortByFileName = (
      a: DocItem & { _id?: string },
      b: DocItem & { _id?: string },
    ): number => {
      const fileNameA = getFileNameFromId(a);
      const fileNameB = getFileNameFromId(b);
      return naturalCompare(fileNameA, fileNameB);
    };
    
    const result: DocLink[] = [];
    
    // 그룹별로 처리
    const groups = Array.from(groupMap.entries()).sort(([a], [b]) => {
      const priorityA = GROUP_PRIORITY.indexOf(a.toLowerCase());
      const priorityB = GROUP_PRIORITY.indexOf(b.toLowerCase());

      // 우선순위 목록에 둘 다 포함된 경우에만 사용자 지정 순서 적용
      if (priorityA !== -1 && priorityB !== -1 && priorityA !== priorityB) {
        return priorityA - priorityB;
      }
      return a.localeCompare(b);
    });
    
    for (const [groupName, items] of groups) {
      const normalizedGroupName = groupName.toLowerCase();
      
      // 정렬 실행: id에서 추출한 파일명 순서대로
      const sortedItems = [...items]
        .sort(sortByFileName)
        .map(({ _id, ...item }) => item);
      
      if (ACCORDION_GROUPS.has(normalizedGroupName)) {
        // 아코디언 그룹 (Alphazero, Minimax)
        result.push({
          label: groupName,
          items: sortedItems,
        });
      } else {
        // 일반 그룹: 그룹 헤더 + 아이템들
        // 그룹 헤더는 단순 텍스트로 표시 (아코디언 아님)
        result.push({
          label: groupName,
          items: sortedItems,
        });
      }
    }
    
    docLinks.value = result;
  };

  return {
    docLinks,
    pathToGroupMap,
    openGroups,
    isGroupOpen,
    toggleGroup,
    addOpenGroup,
    fetchDocLinks,
  };
});
