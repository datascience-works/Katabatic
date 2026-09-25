"use client";

import UploadDataset from "./components/UploadDataset";
import { workspaceUrls } from "./navigation";

export default function Home() {
  const handleSubmit = () => {
    // Frontend preview only: validation happens in UploadDataset; no file is transmitted.
    window.location.assign(workspaceUrls.models);
  };

  return <UploadDataset onSubmit={handleSubmit} />;
}
