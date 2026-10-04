"use client";

import { useEffect, useRef, useState } from "react";
import { workspaceUrls } from "./navigation";
import type { ReactNode } from "react";

// Kept local so each frontend remains independently runnable.
const navigation = [
  { label: "Overview", href: workspaceUrls.dashboard, path: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z" },
  { label: "Datasets", href: workspaceUrls.datasets, path: "M3 5c0-4 18-4 18 0s-18 4-18 0 M3 5v14c0 4 18 4 18 0V5 M3 12c0 4 18 4 18 0" },
  { label: "Models", href: workspaceUrls.models, path: "m12 3 9 5-9 5-9-5 9-5 M3 8v9l9 5 9-5V8 M12 13v9" },
  { label: "Results", href: workspaceUrls.results, path: "M4 3v18h17 M9 16v-5 M14 16V7 M19 16V4" },
];

export default function WorkspaceShell({ active, title, children }: { active: string; title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLButtonElement>(null);

  function closeNavigation() {
    setOpen(false);
    menuRef.current?.focus();
  }

  useEffect(() => {
    if (!open) return;
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setOpen(false);
        menuRef.current?.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open]);

  return (
    <div className="workspace-shell">
      <a className="workspace-skip" href="#workspace-content">Skip to content</a>
      <aside id="workspace-navigation" className={`workspace-sidebar ${open ? "is-open" : ""}`}>
        <div className="workspace-brand"><span className="workspace-mark" aria-hidden="true">✦</span><span>katabatic</span><button className="workspace-close" onClick={closeNavigation} aria-label="Close navigation">×</button></div>
        <nav aria-label="Main navigation">
          <p className="workspace-nav-label">Workspace</p>
          {navigation.map((item) => (
            <a key={item.label} href={active === item.label ? "#workspace-content" : item.href} className={`workspace-nav-item ${active === item.label ? "active" : ""}`} aria-current={active === item.label ? "page" : undefined} onClick={() => setOpen(false)}>
              <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={item.path} /></svg>
              <span>{item.label}</span>
            </a>
          ))}
        </nav>
        <div className="workspace-profile"><span className="workspace-avatar" aria-hidden="true">K</span><div><strong>Katabatic workspace</strong><span>Synthetic data platform</span></div></div>
      </aside>
      {open && <button className="workspace-backdrop" aria-label="Close navigation" onClick={closeNavigation} />}
      <div className="workspace-main">
        <header className="workspace-topbar">
          <button ref={menuRef} className="workspace-menu" aria-label="Open navigation" aria-expanded={open} aria-controls="workspace-navigation" onClick={() => setOpen(!open)}><svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16" /></svg></button>
          <div className="workspace-breadcrumb"><span>Workspace</span><span aria-hidden="true">/</span><strong>{title}</strong></div>
          <span className="workspace-context">Synthetic data workspace</span>
        </header>
        <main id="workspace-content" tabIndex={-1}>{children}</main>
      </div>
    </div>
  );
}
