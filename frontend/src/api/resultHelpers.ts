export function readNumber(
  values: Record<string, unknown>,
  key: string,
  fallback = 0,
): number {
  const value = values[key]

  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : fallback
}

export function readBoolean(
  values: Record<string, unknown>,
  key: string,
  fallback = false,
): boolean {
  const value = values[key]

  return typeof value === 'boolean' ? value : fallback
}

export function readString(
  values: Record<string, unknown>,
  key: string,
): string | null {
  const value = values[key]

  return typeof value === 'string' && value.length > 0 ? value : null
}

export function formatEvidence(value: unknown): string {
  if (value === null || value === undefined) {
    return 'Not provided'
  }

  if (typeof value === 'string') {
    return value
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return 'Unable to display evidence'
  }
}