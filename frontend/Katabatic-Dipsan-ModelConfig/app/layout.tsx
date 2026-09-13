import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Model Configuration | Katabatic",
  description: "Configure a synthetic data training experiment",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
