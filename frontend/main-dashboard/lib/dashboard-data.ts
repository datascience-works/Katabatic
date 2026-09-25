export type Dataset = {
  id: string;
  name: string;
  rows: number;
  columns: number;
  size: string;
  updated: string;
  status: "Ready" | "Processing";
};

export type Model = {
  id: string;
  name: string;
  family: string;
  version: string;
  status: "Available" | "Experimental";
};

export type Experiment = {
  id: string;
  name: string;
  dataset: string;
  model: string;
  score: number;
  status: "Completed" | "Training" | "Queued";
  updated: string;
};

export type Activity = {
  id: string;
  title: string;
  detail: string;
  time: string;
  tone: "success" | "active" | "neutral";
};

export const datasets: Dataset[] = [
  { id: "ds-001", name: "Adult Census", rows: 48842, columns: 15, size: "4.2 MB", updated: "2 min ago", status: "Ready" },
  { id: "ds-002", name: "Bank Marketing", rows: 45211, columns: 17, size: "5.8 MB", updated: "Yesterday", status: "Ready" },
  { id: "ds-003", name: "Credit Card", rows: 30000, columns: 24, size: "7.1 MB", updated: "3 days ago", status: "Ready" },
  { id: "ds-004", name: "MAGIC Telescope", rows: 19020, columns: 11, size: "2.4 MB", updated: "5 days ago", status: "Ready" },
];

export const models: Model[] = [
  { id: "model-001", name: "CTGAN", family: "GAN", version: "0.11.0", status: "Available" },
  { id: "model-002", name: "GANBLR++", family: "Bayesian GAN", version: "1.2.0", status: "Available" },
  { id: "model-003", name: "TabDDPM", family: "Diffusion", version: "0.9.1", status: "Available" },
  { id: "model-004", name: "TabSyn", family: "Diffusion", version: "0.4.0", status: "Experimental" },
  { id: "model-005", name: "PATE-GAN", family: "Private GAN", version: "1.0.0", status: "Available" },
  { id: "model-006", name: "CoDi", family: "Diffusion", version: "0.7.2", status: "Experimental" },
];

export const experiments: Experiment[] = [
  { id: "exp-1048", name: "Census privacy run", dataset: "Adult Census", model: "CTGAN", score: 0.91, status: "Completed", updated: "2 min ago" },
  { id: "exp-1047", name: "Bank utility benchmark", dataset: "Bank Marketing", model: "TabDDPM", score: 0.88, status: "Training", updated: "18 min ago" },
  { id: "exp-1046", name: "Credit fidelity test", dataset: "Credit Card", model: "GANBLR++", score: 0.86, status: "Completed", updated: "Yesterday" },
  { id: "exp-1045", name: "Telescope baseline", dataset: "MAGIC Telescope", model: "PATE-GAN", score: 0.79, status: "Queued", updated: "Yesterday" },
];

export const activity: Activity[] = [
  { id: "activity-1", title: "Training completed", detail: "Census privacy run · CTGAN", time: "2 min ago", tone: "success" },
  { id: "activity-2", title: "Training in progress", detail: "Bank utility benchmark · 68%", time: "18 min ago", tone: "active" },
  { id: "activity-3", title: "Dataset validated", detail: "Credit Card · 30,000 rows", time: "Yesterday", tone: "neutral" },
  { id: "activity-4", title: "Experiment queued", detail: "Telescope baseline · PATE-GAN", time: "Yesterday", tone: "neutral" },
];

export const weeklyRuns = [12, 18, 15, 25, 21, 34, 30, 42, 38, 48, 44, 57];
