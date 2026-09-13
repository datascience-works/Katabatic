import './App.css'

function App() {
  return (
    <div className="results-dashboard">

      <div className="dashboard-header">
        <div>
          <p className="eyebrow">EVALUATION RESULTS</p>
          <h1>Results Dashboard</h1>

          <p className="subtitle">
            Your synthetic dataset is ready. Review the quality scores,
            compare model runs, and download the outputs you need.
          </p>
        </div>

        <div className="status-badge">
          Completed successfully
        </div>
      </div>

      <div className="experiment-summary">
        <div>
          <span>EXPERIMENT</span>
          <strong>Customer churn study</strong>
        </div>

        <div>
          <span>SELECTED MODEL</span>
          <strong>CTGAN V1</strong>
        </div>

        <div>
          <span>TASK</span>
          <strong>Binary classification</strong>
        </div>

        <div>
          <span>DATASET</span>
          <strong>telecom_churn.csv</strong>
        </div>

        <div>
          <span>RUN ID</span>
          <strong>KT-240801-07</strong>
        </div>
      </div>

      <div className="results-overview">

        <div className="score-card">
          <p className="card-label">
            COMPOSITE QUALITY SCORE
          </p>

          <div className="score-value">
            87 <span>/ 100</span>
          </div>

          <div className="score-badge">
            Excellent
          </div>

          <p className="score-description">
            Strong overall quality. The synthetic data preserves important
            patterns while maintaining high privacy protection.
          </p>

          <div className="score-bar">
            <div className="score-bar-fill"></div>
          </div>

          <div className="score-status">
            ✓ All 6 dimensions evaluated
          </div>
        </div>

        <div className="dataset-card">
          <div className="dataset-card-header">
            <div>
              <h2>Your dataset is ready</h2>
              <p>
                Generated successfully and ready for downstream analysis.
              </p>
            </div>

            <span className="ready-badge">
              READY
            </span>
          </div>

          <div className="dataset-stats">
            <div>
              <span>ROWS</span>
              <strong>42,100</strong>
              <small>Synthetic records</small>
            </div>

            <div>
              <span>COLUMNS</span>
              <strong>18</strong>
              <small>Target: churned</small>
            </div>

            <div>
              <span>FILE SIZE</span>
              <strong>8.6 MB</strong>
              <small>CSV output</small>
            </div>

            <div>
              <span>COMPLETED</span>
              <strong>1:56 pm</strong>
              <small>16 min 54 sec</small>
            </div>
          </div>
        </div>

      </div>

      <div className="quality-section">

        <div className="quality-header">
          <div>
            <h2>Quality breakdown</h2>
            <p>
              All six evaluation dimensions are shown below.
              Higher scores mean stronger results.
            </p>
          </div>

          <span className="quality-summary-badge">
            80+ = strong result
          </span>
        </div>

        <div className="quality-grid">

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Utility</h3>
              <span className="metric-label excellent">Excellent</span>
            </div>

            <strong>
              91 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '91%' }}
              ></div>
            </div>

            <p>Predictive usefulness on real data.</p>
          </div>

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Fidelity</h3>
              <span className="metric-label strong">Strong</span>
            </div>

            <strong>
              88 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '88%' }}
              ></div>
            </div>

            <p>Similarity to original distributions.</p>
          </div>

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Privacy</h3>
              <span className="metric-label excellent">Excellent</span>
            </div>

            <strong>
              94 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '94%' }}
              ></div>
            </div>

            <p>Resistance to re-identification.</p>
          </div>

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Diversity</h3>
              <span className="metric-label strong">Strong</span>
            </div>

            <strong>
              84 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '84%' }}
              ></div>
            </div>

            <p>Coverage of realistic data patterns.</p>
          </div>

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Fairness</h3>
              <span className="metric-label good">Good</span>
            </div>

            <strong>
              82 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '82%' }}
              ></div>
            </div>

            <p>Balanced outcomes across groups.</p>
          </div>

          <div className="quality-card">
            <div className="quality-card-top">
              <h3>Robustness</h3>
              <span className="metric-label good">Good</span>
            </div>

            <strong>
              80 <span>/ 100</span>
            </strong>

            <div className="metric-bar">
              <div
                className="metric-fill"
                style={{ width: '80%' }}
              ></div>
            </div>

            <p>Stability across repeated tests.</p>
          </div>

        </div>
      </div>

      <div className="compare-section">

        <div className="compare-header">
          <div>
            <h2>Compare model runs</h2>
            <p>
              Select two or more completed runs to see their evaluation
              scores side by side.
            </p>
          </div>

          <span className="best-model-badge">
            Best overall - CTGAN V1
          </span>
        </div>

        <div className="model-tabs">
          <button className="model-tab active">
            CTGAN V1
          </button>

          <button className="model-tab">
            TVAE V2
          </button>

          <button className="model-tab">
            Gaussian Copula
          </button>

          <button className="model-tab add-run">
            + Add completed run
          </button>
        </div>

        <div className="comparison-table">

          <div className="comparison-row comparison-heading">
            <span>Dimension</span>
            <span>CTGAN V1</span>
            <span>TVAE V2</span>
            <span>Gaussian Copula</span>
          </div>

          <div className="comparison-row">
            <span>Utility</span>
            <span>91</span>
            <span>86</span>
            <span>78</span>
          </div>

          <div className="comparison-row">
            <span>Fidelity</span>
            <span>88</span>
            <span>85</span>
            <span>80</span>
          </div>

          <div className="comparison-row">
            <span>Privacy</span>
            <span>94</span>
            <span>92</span>
            <span>91</span>
          </div>

          <div className="comparison-row">
            <span>Diversity</span>
            <span>84</span>
            <span>86</span>
            <span>76</span>
          </div>

          <div className="comparison-row">
            <span>Fairness</span>
            <span>82</span>
            <span>79</span>
            <span>74</span>
          </div>

          <div className="comparison-row">
            <span>Robustness</span>
            <span>80</span>
            <span>78</span>
            <span>75</span>
          </div>

        </div>
      </div>
         <div className="run-status-section">
  <div className="run-status-header">
    <div>
      <h2>Recent run status</h2>
      <p>
        Downloads are available only when a run finishes successfully and
        produces valid output.
      </p>
    </div>

    <span className="run-count-badge">3 recent runs</span>
  </div>

  <div className="run-list">

    <div className="run-card">
      <div>
        <span className="run-badge completed">Completed</span>
        <h3>Customer churn study</h3>
        <p>CTGAN V1 · KT-240801-07</p>
      </div>

      <div className="run-details">
        <strong>Composite score 87 · 42,100 valid rows</strong>
        <p>Completed at 1:56 pm in 16 min 54 sec.</p>
      </div>

      <div className="run-actions">
        <button>Download CSV</button>
        <button>Download report</button>
      </div>
    </div>

    <div className="run-card">
      <div>
        <span className="run-badge progress">In progress</span>
        <h3>Churn sensitivity test</h3>
        <p>TVAE V2 · KT-240801-09</p>
      </div>

      <div className="run-details">
        <strong>Generating records</strong>
        <p>62% complete · about 9 min left</p>
      </div>

      <div className="run-actions">
        <button disabled>Download CSV</button>
        <button>View progress</button>
      </div>
    </div>

    <div className="run-card">
      <div>
        <span className="run-badge failed">Failed</span>
        <h3>High-detail benchmark</h3>
        <p>CTAB-GAN+ · KT-240801-04</p>
      </div>

      <div className="run-details">
        <strong>Training stopped: GPU memory limit exceeded</strong>
        <p>Retry with a smaller batch size or larger resource profile.</p>
      </div>

      <div className="run-actions">
        <button disabled>Download CSV</button>
        <button>Retry with batch size 512</button>
      </div>
    </div>

  </div>
</div>

<div className="final-action-section">
  <div>
    <h3>You&apos;re ready to use this synthetic dataset</h3>
    <p>
      The selected CTGAN run completed successfully and all evaluation
      dimensions have been reviewed.
    </p>
  </div>

  <div className="final-actions">
    <button className="secondary-action">
      Back to experiments
    </button>

    <button className="primary-action">
      Download dataset
    </button>
  </div>
</div>
    </div>
  )
}

export default App