"use client";

import UploadDataset from "./components/UploadDataset";

export default function Home() {
  const handleSubmit = async (file: File) => {
    // Wired up to the real API in the next step.
    console.log("Selected file ready to upload:", file.name);
  };

  return <UploadDataset onSubmit={handleSubmit} />;
}
