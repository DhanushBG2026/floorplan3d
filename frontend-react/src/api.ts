import type { PlanModel } from './types'

export interface ConvertOptions {
  drawingScale: number
  knownWidthM: number
  pxPerM: number
  crop?: [number, number, number, number] | null
}

export async function convertPlan(file: File, o: ConvertOptions): Promise<PlanModel> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('drawing_scale', String(o.drawingScale || 100))
  fd.append('known_width_m', String(o.knownWidthM || 0))
  fd.append('px_per_m', String(o.pxPerM || 0))
  if (o.crop) fd.append('crop', o.crop.join(','))

  const res = await fetch('/api/convert', { method: 'POST', body: fd })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export interface PreviewResult { image: string; width: number; height: number; pages: number; text: string}

export interface ChatMessage { role: 'user' | 'assistant'; content: string; error?: boolean }

export async function chatWithDocument(documentText: string, messages: ChatMessage[]): Promise<string> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      document: documentText,
      messages: messages.filter(m => !m.error).map(({ role, content }) => ({ role, content })),
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText)
  }
  const data = await res.json()
  return data.reply as string
}

export async function previewPlan(file: File): Promise<PreviewResult> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch('/api/preview', { method: 'POST', body: fd })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}