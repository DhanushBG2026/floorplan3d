import type { PlanModel } from './types'

export interface ImageAdjustments {
  brightness: number
  contrast: number
  highlights: number
  shadows: number
  whites: number
  blacks: number
  sharpness: number
  grayscale: boolean
  invert: boolean
}

export const DEFAULT_ADJUSTMENTS: ImageAdjustments = {
  brightness: 0,
  contrast: 0,
  highlights: 0,
  shadows: 0,
  whites: 0,
  blacks: 0,
  sharpness: 0,
  grayscale: false,
  invert: false,
}

export interface ConvertOptions {
  drawingScale: number
  knownWidthM: number
  pxPerM: number
  crop?: [number, number, number, number] | null
  adjustments?: ImageAdjustments
  detector?: 'opencv' | 'ml'
}

export async function convertPlan(file: File, o: ConvertOptions): Promise<PlanModel> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('detector', o.detector ?? 'opencv')
  fd.append('drawing_scale', String(o.drawingScale || 100))
  fd.append('known_width_m', String(o.knownWidthM || 0))
  fd.append('px_per_m', String(o.pxPerM || 0))
  if (o.crop) fd.append('crop', o.crop.join(','))
  if (o.adjustments) {
    Object.entries(o.adjustments).forEach(([key, value]) => {
      fd.append(key, String(value))
    })

  const a = o.adjustments
  if (a) {
    fd.append('brightness', String(a.brightness))
    fd.append('contrast', String(a.contrast))
    fd.append('highlights', String(a.highlights))
    fd.append('shadows', String(a.shadows))
    fd.append('whites', String(a.whites))
    fd.append('blacks', String(a.blacks))
    fd.append('sharpness', String(a.sharpness))
    fd.append('grayscale', String(a.grayscale))
    fd.append('invert', String(a.invert))
  }
  }

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