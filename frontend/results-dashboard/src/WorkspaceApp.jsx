import { useEffect } from 'react'
import Dashboard from '../../Katabatic-Manya-Dashboard/app/page'
import UploadPage from '../../Katabatic-Bright-UploadDataset/app/page'
import UploadShell from '../../Katabatic-Bright-UploadDataset/app/WorkspaceShell'
import { ModelConfiguration } from '../../Katabatic-Dipsan-ModelConfig/components/model-configuration'
import Results from './App'

// Inline CSS is mounted with the active page, preserving each page's styling
// without loading conflicting global selectors from the other three apps.
import dashboardCss from '../../Katabatic-Manya-Dashboard/app/globals.css?inline'
import uploadCss from '../../Katabatic-Bright-UploadDataset/app/globals.css?inline'
import uploadShellCss from '../../Katabatic-Bright-UploadDataset/app/workspace.css?inline'
import modelCss from '../../Katabatic-Dipsan-ModelConfig/app/globals.css?inline'
import modelShellCss from '../../Katabatic-Dipsan-ModelConfig/app/workspace.css?inline'
import resultsBaseCss from './index.css?inline'
import resultsCss from './App.css?inline'
import resultsShellCss from './workspace.css?inline'

const pages = {
  '/': { title: 'Dashboard', css: dashboardCss, content: <Dashboard /> },
  '/datasets': {
    title: 'Upload Dataset',
    css: uploadCss + uploadShellCss,
    content: <UploadShell active="Datasets" title="Upload dataset"><UploadPage /></UploadShell>,
  },
  '/models': {
    title: 'Model Configuration',
    css: modelCss + modelShellCss,
    content: <ModelConfiguration />,
  },
  '/results': {
    title: 'Results Dashboard',
    css: resultsBaseCss + resultsCss + resultsShellCss,
    content: <Results />,
  },
}

export default function WorkspaceApp() {
  // Standard links give every page a bookmarkable URL and native history.
  // Vite's SPA fallback also serves these routes on a direct visit or refresh.
  const path = window.location.pathname.replace(/\/+$/, '') || '/'
  const page = pages[path]

  useEffect(() => {
    document.title = `${page?.title || 'Page not found'} | Katabatic`
  }, [page])

  if (!page) {
    return <><style>{resultsBaseCss}</style><main style={{ padding: 32 }}><h1>Page not found</h1><a href="/">Return to dashboard</a></main></>
  }

  return <><style>{page.css}</style>{page.content}</>
}
