"use client";

import { useMemo, useState } from "react";

type FieldProps = {
  label: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
};

type SchemaRow = {
  name: string;
  detail: string;
  type: "Continuous" | "Categorical";
  constraint: string;
};

const schemaRows: SchemaRow[] = [
  { name: "customer_age", detail: "Integer · 48% unique", type: "Continuous", constraint: "Min 18 · Max 79" },
  { name: "annual_income", detail: "Decimal · 38% nulls", type: "Continuous", constraint: "Min $500 · Max $250,000" },
  { name: "segment", detail: "4 unique values", type: "Categorical", constraint: "Allowed: Standard, Plus, Pro…" },
  { name: "churned", detail: "2 unique values", type: "Categorical", constraint: "Allowed: Yes, No" },
  { name: "region", detail: "4 unique values", type: "Categorical", constraint: "All observed values" },
];

function Field({ label, hint, children, className = "" }: FieldProps) {
  return (
    <label className={`field ${className}`}>
      <span className="field-label">{label}</span>
      {hint && <span className="field-hint">{hint}</span>}
      {children}
    </label>
  );
}

function Section({ number, title, description, children }: { number: number; title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="card form-section">
      <div className="section-heading">
        <h2>{number}. {title}</h2>
        <p>{description}</p>
      </div>
      {children}
    </section>
  );
}

function Workflow() {
  const steps = [
    ["✓", "Dataset", "customers_q2.csv", "done"],
    ["✓", "Model Selection", "CTGAN", "done"],
    ["3", "Configuration", "Current step", "active"],
    ["4", "Training", "Not started", "pending"],
  ];
  return (
    <div className="card workflow">
      {steps.map(([number, label, detail, state]) => (
        <div className={`workflow-step ${state}`} key={label}>
          <span className="step-number">{number}</span>
          <span><b>{label}</b><small>{detail}</small></span>
        </div>
      ))}
    </div>
  );
}

function SchemaTable() {
  const [target, setTarget] = useState("churned");
  return (
    <>
      <div className="target-row">
        <Field label="Target column *" hint="Required to enable Start Training.">
          <select value={target} onChange={(event) => setTarget(event.target.value)}>
            {schemaRows.map((row) => <option key={row.name}>{row.name}</option>)}
          </select>
        </Field>
        <div className="target-callout"><b>✓ Target selected</b><span>If correct, training is focused on achieving “churned”.</span></div>
      </div>
      <div className="chips"><span>5 columns</span><span>3 categorical</span><span>2 continuous</span><span>1 target</span></div>
      <div className="schema-table">
        <div className="schema-head"><span>Column</span><span>Type</span><span>Constraints</span><span>Target</span></div>
        {schemaRows.map((row) => (
          <div className={`schema-row ${target === row.name ? "selected" : ""}`} key={row.name}>
            <span><b>{row.name}</b><small>{row.detail}</small></span>
            <select defaultValue={row.type}><option>Continuous</option><option>Categorical</option></select>
            <input defaultValue={row.constraint} aria-label={`${row.name} constraints`} />
            <input type="radio" name="target" checked={target === row.name} onChange={() => setTarget(row.name)} aria-label={`Use ${row.name} as target`} />
          </div>
        ))}
      </div>
      <button className="text-button">+ Add a custom constraint</button>
      <div className="warning"><span>!</span><p><b>Constraint conflicts with observed data</b><small>annual_income excludes 12% of rows below $500. You can keep this draft, but a warning must be acknowledged before training.</small></p></div>
    </>
  );
}

function TrainingParameters() {
  const [earlyStopping, setEarlyStopping] = useState(true);
  return (
    <Section number={3} title="Training parameters" description="Sensible CTGAN defaults are pre-filled. Every value remains editable.">
      <div className="recommendation"><span>✦</span><p><b>Recommended defaults for CTGAN</b><small>Optimised for a 42k-row tabular dataset. Edit any parameter before starting.</small></p><em>Model-specific</em></div>
      <div className="split-title"><b>Dataset split</b><span>Total 100%</span></div>
      <div className="split-bar"><i /><i /><i /></div>
      <div className="form-grid four">
        <Field label="Training split"><input defaultValue="70%" /></Field>
        <Field label="Validation split"><input defaultValue="15%" /></Field>
        <Field label="Test split"><input defaultValue="15%" /></Field>
        <Field label="Random seed"><input defaultValue="42" /></Field>
      </div>
      <div className="form-grid three">
        <Field label="Number of epochs"><input defaultValue="300" /></Field>
        <Field label="Batch size"><input defaultValue="500" /></Field>
        <Field label="Learning rate"><input defaultValue="0.0002" /></Field>
      </div>
      <div className="form-grid two">
        <Field label="Optimiser"><select defaultValue="Adam"><option>Adam</option><option>SGD</option></select></Field>
        <Field label="Loss function"><select defaultValue="Wasserstein + GP"><option>Wasserstein + GP</option><option>Cross entropy</option></select></Field>
      </div>
      <div className="toggle-row"><span><b>Early stopping</b><small>Stop when validation loss does not improve for 20 epochs.</small></span><button onClick={() => setEarlyStopping(!earlyStopping)} className={`toggle ${earlyStopping ? "on" : ""}`} aria-label="Toggle early stopping"><i /></button></div>
    </Section>
  );
}

function AdvancedSettings() {
  const [open, setOpen] = useState(true);
  return (
    <section className="card advanced">
      <button className="advanced-header" onClick={() => setOpen(!open)}><span><b>Advanced settings</b><small>Optional controls for experienced users. Defaults are safe for this model.</small></span><span className="expanded">{open ? "Expanded⌃" : "Collapsed⌄"}</span></button>
      {open && <><div className="form-grid four">
        <Field label="Gradient penalty"><input defaultValue="10" /></Field>
        <Field label="Discriminator steps"><input defaultValue="5" /></Field>
        <Field label="Noise dimension"><input defaultValue="128" /></Field>
        <Field label="Weight decay"><input defaultValue="0" /></Field>
      </div><div className="repro"><span>●</span><p><b>Reproducible run</b><small>Random seed 42 and environment snapshot will be stored with this experiment.</small></p></div></>}
    </section>
  );
}

function SideRail() {
  const readiness = ["Target column selected", "Split totals 100%", "Parameters complete"];
  return (
    <aside className="side-rail">
      <section className="card summary">
        <div className="side-title"><span><b>Configuration summary</b><small>Review before training</small></span><em>1 warning</em></div>
        <dl><div><dt>Dataset</dt><dd>customers_q2.csv <small>42.2k rows</small></dd></div><div><dt>Selected model</dt><dd className="blue">CTGAN <small>Change · 7 available</small></dd></div><div><dt>Task type</dt><dd>Tabular synthesis <small>Binary target</small></dd></div><div><dt>Estimated duration</dt><dd>≈ 18 minutes <small>± 4 min</small></dd></div><div><dt>Resources</dt><dd>1× NVIDIA T4 <small>8 GB RAM · 2 vCPU</small></dd></div></dl>
        <div className="readiness"><b>TRAINING READINESS</b>{readiness.map((item) => <div key={item}><i>✓</i><span>{item}</span><em>Ready</em></div>)}<div className="review"><i>!</i><span>Constraint warning</span><em>Review</em></div></div>
      </section>
      <section className="card catalogue">
        <div className="side-title"><span><b>Model catalogue</b><small>17 selectable models</small></span><em>17+</em></div>
        {["CTGAN · selected", "TVAE", "CopulaGAN", "Gaussian Copula", "CTAB-GAN+", "TabDDPM"].map((model, index) => <div className={index === 0 ? "model selected" : "model"} key={model}><i />{model}{index === 0 && <small>Recommended</small>}</div>)}
        <button className="text-button">+ 11 more models available in Model Selection</button>
      </section>
      <section className="compute"><small>COMPUTE ESTIMATE</small><h3>Balanced</h3><p>This setup fits the standard GPU profile. No extra quota is required.</p><div><span>T4 GPU</span><span>8 GB RAM</span></div></section>
    </aside>
  );
}

export function ModelConfiguration() {
  const [name, setName] = useState("Customer churn — CTGAN v1");
  const [saved, setSaved] = useState(false);
  const status = useMemo(() => saved ? "Saved just now" : "Last saved just now", [saved]);
  return (
    <div className="app-shell">
      <header className="topbar"><a className="brand">Katabatic</a><nav>{["Overview", "Datasets", "Models", "Experiments", "Training", "Results"].map((item) => <a className={item === "Experiments" ? "active" : ""} key={item}>{item}</a>)}</nav><div className="top-actions"><span className="saved">● All changes saved</span><button>? Docs</button><button className="notification">•</button></div></header>
      <main>
        <div className="page-heading"><div><span className="eyebrow">● &nbsp; EXPERIMENT SETUP</span><h1>Model Configuration</h1><p>Prepare your schema, training parameters, and safeguards before the run starts.</p></div><span className="draft">● Draft configuration</span></div>
        <Workflow />
        <div className="content-grid">
          <div className="form-stack">
            <Section number={1} title="Experiment details" description="Name this run so results are easy to find and compare later.">
              <Field label="Experiment name" hint="Use a unique, descriptive name for this run."><input value={name} onChange={(event) => { setName(event.target.value); setSaved(false); }} /></Field>
              <Field label="Experiment description" hint="Optional context for collaborators and future comparisons."><textarea defaultValue="Generate privacy-safe customer records for churn modelling and retention analysis." /></Field>
            </Section>
            <Section number={2} title="Data schema & target" description="Assign every column as categorical or continuous, then select exactly one target."><SchemaTable /></Section>
            <TrainingParameters />
            <AdvancedSettings />
          </div>
          <SideRail />
        </div>
        <footer className="card action-bar"><button className="back">← &nbsp; Back</button><div className="action-spacer" /><span><b>{status}</b><small>Draft can be resumed later</small></span><button onClick={() => setSaved(true)} className="secondary">Save Configuration</button><button className="primary">Start Training &nbsp; →</button></footer>
      </main>
    </div>
  );
}
