import { type ChangeEvent, type FormEvent, useState } from 'react'
import { analyzePolicyDocuments } from './api/policyAnalysis'
import {
  formatEvidence,
  readBoolean,
  readNumber,
  readString,
} from './api/resultHelpers'
import type { AnalysisResult, Finding } from './api/types'
import './App.css'

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.txt']

function App() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? [])

    setSelectedFiles(files)
    setResult(null)
    setError(null)
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (selectedFiles.length === 0) {
      setError('Select at least one policy document.')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)

    try {
      const analysisResult = await analyzePolicyDocuments(selectedFiles)
      setResult(analysisResult)
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'Policy analysis request failed.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="page-header">
        <p className="eyebrow">Enterprise Policy Intelligence</p>
        <h1>Policy Conflict &amp; Staleness Detector</h1>
        <p className="page-description">
          Upload enterprise policy documents to identify deterministic
          conflicts, stale requirements, and policy relationships.
        </p>
      </header>

      <section className="panel upload-panel" aria-labelledby="upload-heading">
        <div>
          <h2 id="upload-heading">Analyze policy documents</h2>
          <p>
            Select one or more PDF, DOCX, or TXT files. Files are sent together
            in a single analysis request.
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <label className="file-picker">
            <span>Select policy documents</span>
            <input
              type="file"
              multiple
              accept={ALLOWED_EXTENSIONS.join(',')}
              onChange={handleFileSelection}
              disabled={isLoading}
            />
          </label>

          {selectedFiles.length > 0 && (
            <div className="selected-files">
              <h3>Selected files ({selectedFiles.length})</h3>
              <ul>
                {selectedFiles.map((file, index) => (
                  <li key={`${file.name}-${file.size}-${index}`}>
                    <span>{file.name}</span>
                    <span>{formatFileSize(file.size)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <button
            className="analyze-button"
            type="submit"
            disabled={isLoading || selectedFiles.length === 0}
          >
            {isLoading ? 'Analyzing policies…' : 'Run policy analysis'}
          </button>
        </form>
      </section>

      {error && (
        <section className="message error-message" role="alert">
          <strong>Analysis failed</strong>
          <span>{error}</span>
        </section>
      )}

      {isLoading && (
        <section className="message loading-message" aria-live="polite">
          <strong>Analysis in progress</strong>
          <span>
            Processing {selectedFiles.length} policy document
            {selectedFiles.length === 1 ? '' : 's'}.
          </span>
        </section>
      )}

      {result && <AnalysisDashboard result={result} />}
    </main>
  )
}

function AnalysisDashboard({ result }: { result: AnalysisResult }) {
  const statistics = result.statistics

  const llmEnabled = readBoolean(statistics, 'llm_enabled')
  const llmProvider = readString(statistics, 'llm_provider')
  const llmModel = readString(statistics, 'llm_model')

  const llmEligible = readNumber(statistics, 'llm_eligible_findings')
  const llmVerified = readNumber(statistics, 'llm_verified_findings')
  const llmRejected = readNumber(statistics, 'llm_rejected_findings')
  const llmBypassed = readNumber(statistics, 'llm_bypassed_findings')
  const llmFailed = readNumber(statistics, 'llm_failed_findings')

  return (
    <section className="results" aria-labelledby="results-heading">
      <div className="results-heading">
        <div>
          <p className="eyebrow">Analysis Result</p>
          <h2 id="results-heading">Policy intelligence dashboard</h2>
        </div>

        <span className="processing-time">
          {result.processing_time_ms.toFixed(1)} ms
        </span>
      </div>

      <div className="metrics-grid">
        <MetricCard label="Policies" value={result.policies.length} />
        <MetricCard label="Obligations" value={result.obligations.length} />
        <MetricCard label="Findings" value={result.findings.length} />
        <MetricCard label="Warnings" value={result.warnings.length} />
      </div>

      <section className="panel observability-panel">
        <div className="section-heading">
          <div>
            <h3>AI verification observability</h3>
            <p>
              Optional LLM verification telemetry returned by the analysis
              engine.
            </p>
          </div>

          <span
            className={`status-badge ${
              llmEnabled ? 'status-enabled' : 'status-disabled'
            }`}
          >
            {llmEnabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>

        <dl className="observability-metadata">
          <div>
            <dt>Provider</dt>
            <dd>{llmProvider ?? 'Not configured'}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd>{llmModel ?? 'Not configured'}</dd>
          </div>
        </dl>

        <div className="metrics-grid compact-metrics">
          <MetricCard label="Eligible" value={llmEligible} />
          <MetricCard label="Verified" value={llmVerified} />
          <MetricCard label="Rejected" value={llmRejected} />
          <MetricCard label="Bypassed" value={llmBypassed} />
          <MetricCard label="Failed" value={llmFailed} />
        </div>
      </section>

      <section className="panel findings-panel">
        <div className="section-heading">
          <div>
            <h3>Policy findings</h3>
            <p>
              Deterministic findings with severity, confidence, evidence, and AI
              verification status.
            </p>
          </div>

          <span className="finding-count">
            {result.findings.length} total
          </span>
        </div>

        {result.findings.length === 0 ? (
          <div className="empty-state">
            <strong>No findings detected</strong>
            <span>
              The analysis completed without returning policy conflicts or stale
              requirements.
            </span>
          </div>
        ) : (
          <div className="findings-list">
            {result.findings.map((finding) => (
              <FindingCard key={finding.finding_id} finding={finding} />
            ))}
          </div>
        )}
      </section>

      {result.warnings.length > 0 && (
        <section className="panel warnings-panel">
          <h3>Analysis warnings</h3>
          <ul>
            {result.warnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </section>
      )}
    </section>
  )
}

function FindingCard({ finding }: { finding: Finding }) {
  const sourceEvidence =
    finding.evidence.source_sentence ??
    finding.evidence.source ??
    finding.evidence.source_text ??
    finding.evidence.source_obligation ??
    'Not provided'

  const targetEvidence =
    finding.evidence.target_sentence ??
    finding.evidence.target ??
    finding.evidence.target_text ??
    finding.evidence.target_obligation ??
    'Not provided'

  return (
    <article className="finding-card">
      <div className="finding-header">
        <div>
          <span className="finding-type">{finding.finding_type}</span>
          <h4>{finding.explanation}</h4>
        </div>

        <span
          className={`severity-badge severity-${finding.severity.toLowerCase()}`}
        >
          {finding.severity}
        </span>
      </div>

      <dl className="finding-metadata">
        <div>
          <dt>Confidence</dt>
          <dd>{formatPercentage(finding.confidence)}</dd>
        </div>
        <div>
          <dt>Deterministic score</dt>
          <dd>{formatPercentage(finding.deterministic_score)}</dd>
        </div>
        <div>
          <dt>AI verification</dt>
          <dd>{finding.llm_verified ? 'Verified' : 'Not verified'}</dd>
        </div>
      </dl>

      <div className="evidence-grid">
        <EvidenceBlock label="Source evidence" value={sourceEvidence} />
        <EvidenceBlock label="Target evidence" value={targetEvidence} />
      </div>

      {finding.evidence.llm_verification !== undefined && (
        <EvidenceBlock
          label="AI verification detail"
          value={finding.evidence.llm_verification}
        />
      )}
    </article>
  )
}

function EvidenceBlock({
  label,
  value,
}: {
  label: string
  value: unknown
}) {
  return (
    <div className="evidence-block">
      <strong>{label}</strong>
      <pre>{formatEvidence(value)}</pre>
    </div>
  )
}

function MetricCard({
  label,
  value,
}: {
  label: string
  value: number
}) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  )
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export default App