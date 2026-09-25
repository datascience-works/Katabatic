"use client";

import React from "react";

const NAV_ITEMS = [
  { label: "Dashboard", href: "#" },
  { label: "Datasets", href: "#", active: true },
  { label: "Models", href: "#" },
  { label: "Experiments", href: "#" },
  { label: "Evaluate", href: "#" },
  { label: "Leaderboard", href: "#" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        Katabatic
      </div>

      <p className="nav-label">Menu</p>
      <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {NAV_ITEMS.map((item) => (
          <a
            key={item.label}
            href={item.href}
            className={`nav-item${item.active ? " active" : ""}`}
          >
            {item.label}
          </a>
        ))}
      </nav>
    </aside>
  );
}