import { useEffect } from 'react'
import Dashboard from '../../main-dashboard/app/page'
import UploadPage from '../../upload-dataset/app/page'
import UploadShell from '../../upload-dataset/app/WorkspaceShell'
import { ModelConfiguration } from '../../model-config/components/model-configuration'
import Results from './App'

// Inline CSS is mounted with the active page, preserving each page's styling
// without loading conflicting global selectors from the other three apps.
import dashboardCss from '../../main-dashboard/app/globals.css?inline'
import uploadCss from '../../upload-dataset/app/globals.css?inline'
import uploadShellCss from '../../upload-dataset/app/workspace.css?inline'
import modelCss from '../../model-config/app/globals.css?inline'
import modelShellCss from '../../model-config/app/workspace.css?inline'
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
