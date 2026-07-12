import type { AnalysisResult } from './types'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

interface ApiErrorResponse {
  detail?: string
}

export async function analyzePolicyDocuments(
  files: File[],
): Promise<AnalysisResult> {
  const formData = new FormData()

  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/analyses`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let message = 'Policy analysis request failed.'

    try {
      const errorBody = (await response.json()) as ApiErrorResponse

      if (errorBody.detail) {
        message = errorBody.detail
      }
    } catch {
      // Keep the safe fallback message when the response is not JSON.
    }

    throw new Error(message)
  }

  return (await response.json()) as AnalysisResult
}