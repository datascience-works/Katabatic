import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Katabatic Dashboard",
  description: "Synthetic data experiments, models and training activity.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
