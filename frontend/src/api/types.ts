export interface PolicySection {
  section_id: string
  title: string | null
  text: string
  paragraphs: string[]
  sentences: string[]
}

export interface NormalizedPolicy {
  policy_id: string
  title: string
  version: string | null
  owner: string | null
  department: string | null
  effective_date: string | null
  review_date: string | null
  status: string | null
  source_file: string
  sections: PolicySection[]
  metadata: Record<string, unknown>
}

export interface NormalizedObligation {
  obligation_id: string
  policy_id: string
  section_id: string
  sentence_text: string
  subject: string | null
  action: string | null
  object: string | null
  technology: string[]
  scope: string | null
  frequency: string | null
  condition: string | null
  exception: string | null
  strength: number
  modality: string
  negated: boolean
  confidence: number
  embedding: number[]
}

export interface Finding {
  finding_id: string
  finding_type: string
  source_obligation_id: string | null
  target_obligation_id: string | null
  severity: string
  confidence: number
  deterministic_score: number
  llm_verified: boolean
  explanation: string
  evidence: Record<string, unknown>
}

export interface AnalysisResult {
  policies: NormalizedPolicy[]
  obligations: NormalizedObligation[]
  findings: Finding[]
  statistics: Record<string, unknown>
  warnings: string[]
  processing_time_ms: number
}