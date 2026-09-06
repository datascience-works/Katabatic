"use client";

import {
  Activity,
  Bell,
  Boxes,
  ChevronDown,
  CircleUserRound,
  Database,
  FlaskConical,
  Gauge,
  Menu,
  Plus,
  Search,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  activity,
  datasets,
  experiments,
  models,
  weeklyRuns,
  type Experiment,
} from "@/lib/dashboard-data";

const navigation = [
  { label: "Overview", icon: Gauge },
  { label: "Datasets", icon: Database },
  { label: "Experiments", icon: FlaskConical },
  { label: "Models", icon: Boxes },
];

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-AU").format(value);
}

function TrendChart() {
  const width = 640;
  const height = 190;
  const max = Math.max(...weeklyRuns);
  const points = weeklyRuns
    .map((value, index) => {
      const x = (index / (weeklyRuns.length - 1)) * width;
      const y = height - (value / max) * (height - 24) - 8;
      return `${x},${y}`;
    })
    .join(" ");
  const area = `0,${height} ${points} ${width},${height}`;

  return (
    <div className="chart" role="img" aria-label="Training runs increased from 12 to 57 over twelve weeks">
      <div className="chart-grid" aria-hidden="true" />
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#6656e8" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#6656e8" stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={area} fill="url(#chart-fill)" />
        <polyline points={points} fill="none" stroke="#6656e8" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        {weeklyRuns.map((value, index) => {
          const x = (index / (weeklyRuns.length - 1)) * width;
          const y = height - (value / max) * (height - 24) - 8;
          return <circle key={`${value}-${index}`} cx={x} cy={y} r="4.5" fill="#ffffff" stroke="#6656e8" strokeWidth="3" />;
        })}
      </svg>
      <div className="chart-labels" aria-hidden="true"><span>Jun</span><span>Jul</span><span>Aug</span><span>Sep</span></div>
    </div>
  );
}

function StatusBadge({ status }: { status: Experiment["status"] }) {
  return <span className={`status status-${status.toLowerCase()}`}>{status}</span>;
}

export default function DashboardPage() {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeNav, setActiveNav] = useState("Overview");
  const [notice, setNotice] = useState("");

  const filteredExperiments = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return experiments;
    return experiments.filter((experiment) =>
      [experiment.name, experiment.dataset, experiment.model, experiment.status]
        .some((value) => value.toLowerCase().includes(needle)),
    );
  }, [query]);

  const announce = (message: string) => {
    setNotice(message);
    window.setTimeout(() => setNotice(""), 2600);
  };

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNavOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <span className="brand-mark"><Sparkles size={20} strokeWidth={2.5} /></span>
          <span>katabatic</span>
        </div>

        <nav aria-label="Main navigation">
          <p className="nav-label">Workspace</p>
          {navigation.map(({ label, icon: Icon }) => (
            <button
              className={`nav-item ${activeNav === label ? "active" : ""}`}
              key={label}
              onClick={() => { setActiveNav(label); setMobileNavOpen(false); announce(`${label} selected`); }}
            >
              <Icon size={19} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <button className="nav-item" onClick={() => announce("Settings selected")}><Settings size={19} /><span>Settings</span></button>
          <div className="profile">
            <span className="avatar">MK</span>
            <div><strong>Manya Khosla</strong><span>Dashboard team</span></div>
            <ChevronDown size={16} />
          </div>
        </div>
      </aside>

      {mobileNavOpen && <button className="backdrop" aria-label="Close navigation" onClick={() => setMobileNavOpen(false)} />}

      <main>
        <header className="topbar">
          <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileNavOpen(true)}><Menu size={21} /></button>
          <label className="search-box">
            <Search size={18} aria-hidden="true" />
            <span className="sr-only">Search experiments</span>
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search experiments, datasets or models" />
            {query && <button aria-label="Clear search" onClick={() => setQuery("")}><X size={16} /></button>}
          </label>
          <div className="top-actions">
            <button className="icon-button notification" aria-label="Notifications" onClick={() => announce("No new notifications")}><Bell size={20} /><span /></button>
            <button className="user-button" onClick={() => announce("Profile menu selected")}><CircleUserRound size={21} /><span>Manya</span><ChevronDown size={15} /></button>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div><p className="eyebrow">SYNTHETIC DATA WORKSPACE</p><h1>Good afternoon, Manya</h1><p>Here’s what’s happening across your Katabatic experiments.</p></div>
            <button className="primary-button" onClick={() => announce("New experiment flow ready to connect")}><Plus size={18} />New experiment</button>
          </section>

          <section className="metrics-grid" aria-label="Workspace summary">
            <article className="metric-card"><div className="metric-top"><span className="metric-icon purple"><Database size={20} /></span><span className="change positive">+2 this month</span></div><strong>{datasets.length}</strong><p>Active datasets</p></article>
            <article className="metric-card"><div className="metric-top"><span className="metric-icon blue"><FlaskConical size={20} /></span><span className="change positive">+18%</span></div><strong>28</strong><p>Total experiments</p></article>
            <article className="metric-card"><div className="metric-top"><span className="metric-icon teal"><Boxes size={20} /></span><span className="change">3 families</span></div><strong>{models.length}</strong><p>Available models</p></article>
            <article className="metric-card"><div className="metric-top"><span className="metric-icon amber"><Activity size={20} /></span><span className="change positive">+4.2%</span></div><strong>86.4%</strong><p>Average quality score</p></article>
          </section>

          <section className="overview-grid">
            <article className="panel chart-panel">
              <div className="panel-heading"><div><h2>Training activity</h2><p>Runs completed over the last 12 weeks</p></div><button onClick={() => announce("Showing the last 12 weeks")}>Last 12 weeks <ChevronDown size={15} /></button></div>
              <TrendChart />
            </article>
            <article className="panel activity-panel">
              <div className="panel-heading"><div><h2>Recent activity</h2><p>Latest workspace updates</p></div></div>
              <div className="activity-list">
                {activity.map((item) => <div className="activity-row" key={item.id}><span className={`activity-dot ${item.tone}`} /><div><strong>{item.title}</strong><p>{item.detail}</p></div><time>{item.time}</time></div>)}
              </div>
            </article>
          </section>

          <section className="panel experiments-panel">
            <div className="panel-heading"><div><h2>Recent experiments</h2><p>{query ? `${filteredExperiments.length} matching results` : "Monitor your latest synthetic data runs"}</p></div><button className="text-button" onClick={() => announce("All experiments selected")}>View all</button></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Experiment</th><th>Dataset</th><th>Model</th><th>Quality score</th><th>Status</th><th>Updated</th></tr></thead>
                <tbody>
                  {filteredExperiments.map((experiment) => <tr key={experiment.id}><td><strong>{experiment.name}</strong><span>{experiment.id}</span></td><td>{experiment.dataset}</td><td><span className="model-chip">{experiment.model}</span></td><td><div className="score-cell"><span className="score-track"><span style={{ width: `${experiment.score * 100}%` }} /></span><strong>{Math.round(experiment.score * 100)}%</strong></div></td><td><StatusBadge status={experiment.status} /></td><td>{experiment.updated}</td></tr>)}
                  {!filteredExperiments.length && <tr><td className="empty-state" colSpan={6}>No experiments match “{query}”.</td></tr>}
                </tbody>
              </table>
            </div>
          </section>

          <footer><span>Katabatic workspace</span><span>{formatNumber(datasets.reduce((total, dataset) => total + dataset.rows, 0))} rows ready for training</span></footer>
        </div>
      </main>
      <div className={`toast ${notice ? "toast-visible" : ""}`} role="status" aria-live="polite">{notice}</div>
    </div>
  );
}
