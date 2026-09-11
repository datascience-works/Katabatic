"use client";

import React, { useCallback, useRef, useState } from "react";

interface UploadDatasetProps {
  /** Called with the validated File once the user clicks Submit */
  onSubmit: (file: File) => void;
}

const ACCEPTED_EXTENSION = ".csv";
const ACCEPTED_MIME_TYPES = ["text/csv", "application/vnd.ms-excel"];
const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB — adjust to match backend limit

function validateFile(file: File): string | null {
  const nameLower = file.name.toLowerCase();
  const hasCsvExtension = nameLower.endsWith(ACCEPTED_EXTENSION);
  const hasCsvMime = file.type === "" || ACCEPTED_MIME_TYPES.includes(file.type);

  if (!hasCsvExtension || !hasCsvMime) {
    return "Dataset not found. Please provide a CSV file.";
  }
  if (file.size === 0) {
    return "The selected file is empty. Please choose a valid CSV file.";
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `File is too large. Maximum size is ${MAX_FILE_SIZE_BYTES / (1024 * 1024)}MB.`;
  }
  return null;
}

export default function UploadDataset({ onSubmit }: UploadDatasetProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File | undefined | null) => {
    if (!file) return;
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      return;
    }
    setError(null);
    setSelectedFile(file);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const onBrowseClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      handleFile(file);
      e.target.value = "";
    },
    [handleFile]
  );

  const onRemoveFile = useCallback(() => {
    setSelectedFile(null);
    setError(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedFile) return;
    setIsSubmitting(true);
    setError(null);
    try {
      onSubmit(selectedFile);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  }, [selectedFile, onSubmit]);

  const dropzoneClassName = [
    "upload-dropzone",
    isDragging ? "dragging" : "",
    error ? "has-error" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="upload-page">
      <div className="upload-heading">
        <h1>Upload Dataset</h1>
        <p>Upload a CSV dataset to begin evaluating synthetic data models.</p>
      </div>

      {error && (
        <div role="alert" className="upload-error-banner">
          <svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM9 9a1 1 0 012 0v3a1 1 0 11-2 0V9zm1-4a1 1 0 100 2 1 1 0 000-2z"
              clipRule="evenodd"
            />
          </svg>
          <span>{error}</span>
        </div>
      )}

      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        className={dropzoneClassName}
      >
        <svg
          className="upload-dropzone-icon"
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 8.25L12 3.75 7.5 8.25M12 3.75v13.5"
          />
        </svg>

        {selectedFile ? (
          <>
            <p className="upload-file-name">{selectedFile.name}</p>
            <p className="upload-file-size">{(selectedFile.size / 1024).toFixed(1)} KB</p>
            <button type="button" onClick={onRemoveFile} className="upload-remove-button">
              Remove file
            </button>
          </>
        ) : (
          <>
            <p>
              Drag &amp; drop your CSV file here <span className="or">OR</span>
            </p>
            <button type="button" onClick={onBrowseClick} className="upload-browse-button">
              Browse Files
            </button>
            <p className="upload-supported-format">Supported format: CSV</p>
          </>
        )}

        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: "none" }}
          onChange={onInputChange}
        />
      </div>

      <div className="upload-actions">
        <button
          type="button"
          disabled={!selectedFile || isSubmitting}
          onClick={handleSubmit}
          className="upload-submit-button"
        >
          {isSubmitting ? "Uploading..." : "Submit"}
        </button>
      </div>
    </div>
  );
}
