import { useEffect, useMemo, useState } from 'react'
import './App.css'

const navItems = [
  'Dashboard',
  'Satellite Image Upload',
  'Water Segmentation',
  'Water Quality Prediction',
  'Spectral Analysis',
  'Temporal Monitoring',
  'Model Evaluation',
  'Results and Reports',
  'About',
]

const blankMetrics = {
  project_name: 'Satellite Water Monitoring AI',
  canonical_metrics: [],
  segmentation_metrics: [],
  latest_stage3_metrics: [],
}

const targetUnits = {
  turbidity: 'FNU',
  chlorophyll: 'μg/L',
  cdom: 'μg/L',
}

function App() {
  const [activePage, setActivePage] = useState('Dashboard')
  const [projectInfo, setProjectInfo] = useState(blankMetrics)
  const [health, setHealth] = useState({ status: 'unknown', model_availability: {} })
  const [uploading, setUploading] = useState(false)
  const [uploadMessage, setUploadMessage] = useState('')
  const [segmentationResult, setSegmentationResult] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [projectRes, healthRes] = await Promise.all([
          fetch('/api/project-info'),
          fetch('/api/health'),
        ])

        if (!projectRes.ok || !healthRes.ok) {
          throw new Error('Unable to reach the backend API.')
        }

        const project = await projectRes.json()
        const healthData = await healthRes.json()
        setProjectInfo(project)
        setHealth(healthData)
      } catch (err) {
        setError(err.message)
      }
    }

    fetchData()
  }, [])

  const metricsByTarget = useMemo(() => {
    const rows = projectInfo.canonical_metrics || []
    return rows.reduce((acc, row) => {
      const target = String(row.target || row['Target'] || '').toLowerCase()
      acc[target] = row
      return acc
    }, {})
  }, [projectInfo])

  const handleSegmentationUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setUploadMessage('')
    setError('')

    try {
      const form = new FormData()
      form.append('file', file)
      const response = await fetch('/api/segmentation', {
        method: 'POST',
        body: form,
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Segmentation failed.')
      }
      setSegmentationResult(data.result)
      setUploadMessage(`Processed ${file.name} successfully.`)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleWqPrediction = async () => {
    setError('')
    try {
      const payload = {
        target: 'turbidity',
        features: {
          ndwi: 0.2,
          mndwi: 0.1,
          ndci: 0.25,
          turbidity_ratio: 1.7,
          cdom_ratio: 0.9,
          b8_b4_ratio: 1.1,
          B01_center_refl: 0.12,
          B02_center_refl: 0.15,
          B03_center_refl: 0.18,
          B04_center_refl: 0.21,
          B05_center_refl: 0.19,
          B06_center_refl: 0.2,
          B07_center_refl: 0.28,
          B08_center_refl: 0.34,
          'B8A_center_refl': 0.3,
          B11_center_refl: 0.22,
          B12_center_refl: 0.18,
        },
      }
      const response = await fetch('/api/water-quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.detail || 'Prediction failed.')
      }
      setPrediction(data.prediction)
    } catch (err) {
      setError(err.message)
    }
  }

  const renderPage = () => {
    switch (activePage) {
      case 'Dashboard':
        return (
          <div className="page-grid">
            <div className="card highlight">
              <span className="eyebrow">System status</span>
              <h3>{health.status === 'ok' ? 'Backend ready' : 'Backend unavailable'}</h3>
              <p>{health.status === 'ok' ? 'FastAPI health endpoint responded successfully.' : 'Waiting for API health check.'}</p>
            </div>
            <div className="card">
              <span className="eyebrow">Segmentation</span>
              <h3>{health.model_availability?.segmentation ? 'Checkpoint available' : 'Checkpoint missing'}</h3>
              <p>Water U-Net model availability check.</p>
            </div>
            <div className="card">
              <span className="eyebrow">Water quality</span>
              <h3>{health.model_availability?.water_quality_dataset ? 'Dataset loaded' : 'Dataset missing'}</h3>
              <p>Matched Sentinel-2 / USGS dataset availability.</p>
            </div>
            <div className="card wide">
              <span className="eyebrow">Canonical benchmark</span>
              <h3>Best reported performance</h3>
              <div className="table-rows">
                {projectInfo.canonical_metrics?.length ? (
                  projectInfo.canonical_metrics.map((row) => (
                    <div className="metric-row" key={row.target || row.Target || Math.random()}>
                      <span>{row.target || row.Target}</span>
                      <strong>{row.best_model || 'Model'}</strong>
                      <em>{row.macro_log_r2 ? `Macro log R² = ${Number(row.macro_log_r2).toFixed(4)}` : 'Metric available'}</em>
                    </div>
                  ))
                ) : (
                  <div className="metric-row"><span>Loading benchmark metrics…</span></div>
                )}
              </div>
            </div>
          </div>
        )
      case 'Satellite Image Upload':
        return (
          <div className="card wide">
            <span className="eyebrow">Upload analysis</span>
            <h3>Upload a multispectral image</h3>
            <input type="file" accept=".tif,.tiff" onChange={handleSegmentationUpload} />
            {uploading && <p className="status">Processing image…</p>}
            {uploadMessage && <p className="status success">{uploadMessage}</p>}
            {error && <p className="status error">{error}</p>}
          </div>
        )
      case 'Water Segmentation':
        return (
          <div className="card wide">
            <span className="eyebrow">Segmentation</span>
            <h3>Water extraction output</h3>
            {segmentationResult ? (
              <div className="results-block">
                <p>Water fraction: {Number(segmentationResult.water_fraction).toFixed(4)}</p>
                <p>Mask shape: {segmentationResult.mask_shape?.join(' × ')}</p>
                <pre>{JSON.stringify(segmentationResult.band_summary, null, 2)}</pre>
              </div>
            ) : (
              <p>Upload a supported GeoTIFF to generate a water mask.</p>
            )}
          </div>
        )
      case 'Water Quality Prediction':
        return (
          <div className="card wide">
            <span className="eyebrow">Inversion</span>
            <h3>Predict water quality</h3>
            <button type="button" className="primary-button" onClick={handleWqPrediction}>Run prediction</button>
            {prediction && (
              <div className="results-block">
                <p>Target: {prediction.target}</p>
                <p>Model: {prediction.model_name}</p>
                <p>Raw prediction: {Number(prediction.raw_prediction).toFixed(4)} {targetUnits[prediction.target]}</p>
                <p>Log prediction: {Number(prediction.log_prediction).toFixed(4)}</p>
              </div>
            )}
            {error && <p className="status error">{error}</p>}
          </div>
        )
      case 'Spectral Analysis':
        return (
          <div className="card wide">
            <span className="eyebrow">Indices</span>
            <h3>Supported spectral features</h3>
            <ul className="feature-list">
              <li>NDWI = (B03 − B08) / (B03 + B08)</li>
              <li>MNDWI = (B03 − B11) / (B03 + B11)</li>
              <li>NDCI = (B05 − B04) / (B05 + B04)</li>
              <li>B4/B3 ratio</li>
              <li>B2/B3 ratio</li>
              <li>B8/B4 ratio</li>
            </ul>
          </div>
        )
      case 'Temporal Monitoring':
        return (
          <div className="card wide">
            <span className="eyebrow">Change tracking</span>
            <h3>Compare two scenes</h3>
            <p>Upload two 6-band TIFFs to estimate water-area change between dates.</p>
          </div>
        )
      case 'Model Evaluation':
        return (
          <div className="page-grid">
            <div className="card wide">
              <span className="eyebrow">Metrics</span>
              <h3>Model evaluation overview</h3>
              <p className="metric-note">
                The values below are raw-scale R² from the latest Stage 3 benchmark. A negative R² is valid:
                it means the model predicts worse than a baseline that always predicts the test-set mean.
                It is not a software error and it should not be compared directly with the canonical log-space R².
              </p>
              <div className="table-rows">
                {projectInfo.latest_stage3_metrics?.length ? projectInfo.latest_stage3_metrics.map((row) => (
                  <div className="metric-row" key={`${row.model}-${row.target}`}>
                    <span>{row.target}</span>
                    <strong>{row.model}</strong>
                    <em>Raw-scale R² = {Number(row.r2 ?? row['R²'] ?? 0).toFixed(4)}</em>
                  </div>
                )) : <div className="metric-row"><span>Latest evaluation metrics unavailable.</span></div>}
              </div>
            </div>
            <div className="card wide">
              <span className="eyebrow">Canonical research benchmark</span>
              <h3>Grouped-validation log-space results</h3>
              <p className="metric-note">
                These are the project&apos;s canonical macro log10 R² values. Negative values indicate poor
                explanatory performance under station-grouped validation, especially for Chlorophyll-a and CDOM.
              </p>
              <div className="table-rows">
                {projectInfo.canonical_metrics?.length ? projectInfo.canonical_metrics.map((row) => (
                  <div className="metric-row" key={`canonical-${row.target}`}>
                    <span>{row.target}</span>
                    <strong>{row.best_model}</strong>
                    <em>Macro log10 R² = {Number(row.macro_log_r2).toFixed(4)}</em>
                  </div>
                )) : <div className="metric-row"><span>Canonical metrics unavailable.</span></div>}
              </div>
            </div>
          </div>
        )
      case 'Results and Reports':
        return (
          <div className="card wide">
            <span className="eyebrow">Outputs</span>
            <h3>Generated reports and tables</h3>
            <div className="report-list">
              {['results/FINAL_RESULTS/final_metrics_summary.csv', 'results/tables/stage3_consolidated_inversion_benchmarks.csv', 'reports/latest_stage3_results_audit.md', 'reports/temporal_matchup_diagnostic_report.md'].map((path) => (
                <div key={path} className="report-item">{path}</div>
              ))}
            </div>
          </div>
        )
      case 'About':
        return (
          <div className="card wide">
            <span className="eyebrow">Research overview</span>
            <h3>Satellite Water Monitoring AI</h3>
            <p>This platform integrates Sentinel-2 water extraction and water-quality inversion for turbidity, chlorophyll-a, and CDOM using research-grade station-grouped validation.</p>
            <p>Important: performance metrics are historical evaluation results and should not be interpreted as universal accuracy for every uploaded image.</p>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">SW</div>
          <div>
            <strong>Satellite Water</strong>
            <small>Monitoring AI</small>
          </div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button
              key={item}
              type="button"
              className={item === activePage ? 'nav-item active' : 'nav-item'}
              onClick={() => setActivePage(item)}
            >
              {item}
            </button>
          ))}
        </nav>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <span className="eyebrow">Environmental monitoring dashboard</span>
            <h1>{activePage}</h1>
          </div>
          <div className="status-pill">{health.status === 'ok' ? 'Online' : 'Offline'}</div>
        </header>

        {error && <div className="banner error">{error}</div>}
        {renderPage()}
      </main>
    </div>
  )
}

export default App
