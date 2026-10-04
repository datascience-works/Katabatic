import type { Metadata } from "next";
import "./globals.css";
import WorkspaceShell from "./WorkspaceShell";
import "./workspace.css";

export const metadata: Metadata = {
  title: "Katabatic",
  description: "Synthetic data evaluation platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <WorkspaceShell active="Datasets" title="Upload dataset">{children}</WorkspaceShell>
      </body>
    </html>
  );
}