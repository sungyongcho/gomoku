import { groupBy, pipe, entries, map, toArray } from "@fxts/core";
import type { DocLink } from "~/types/docs";

export const useDocsStore = defineStore("docs", () => {
  const docLinks = ref<DocLink[]>([]);
  const fetchDocLinks = async () => {
    const docs = await queryCollection("docs").all();

    docLinks.value = pipe(
      docs,
      groupBy((doc) => doc.group),
      entries,
      map(([group, docs]) => ({
        label: group,
        items: docs.map((doc) => ({
          label: doc.title,
          icon: doc.icon || "pi-file",
          url: doc.path,
        })),
      })),
      toArray,
    );
  };

  return {
    docLinks,
    fetchDocLinks,
  };
});
