export type DocItem = {
  label: string;
  icon: string;
  url: string;
};

export type DocFolder = {
  label: string;
  icon?: string;
  items: DocItem[];
};

export type DocLink = DocItem | DocFolder;
