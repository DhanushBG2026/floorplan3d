import type { Wall } from './types'
import type { WallGroups } from './groups'

// How far each end of a wall has been pulled out (metres). Negative shortens the wall.
export interface Ext { start: number; end: number }

export interface AxisInfo {
  horizontal: boolean
  diagonal: boolean
  perp: number          // centre line: z for horizontal walls, x for vertical walls
  thickness: number
  loBase: number        // the two ends along the axis, before any extension
  hiBase: number
  startId: number       // wall piece that carries the start end
  endId: number         // wall piece that carries the far end
  startLen: number      // length of those pieces: limits how far an end can be shortened
  endLen: number
  ax?: number; ay?: number; bx?: number; by?: number // diagonal walls only
}

const MIN_PIECE = 0.1
const MIN_LENGTH = 0.3
const MAX_EXTEND = 30

// A wall with windows or doors is several pieces. Only its two outer ends can be extended.
export function groupAxis(walls: Wall[], groups: WallGroups, gid: number): AxisInfo | null {
  const ids = groups.members[gid]
  if (!ids || ids.length === 0) return null
  const pieces = ids.map(id => walls.find(w => w.id === id)).filter((w): w is Wall => !!w)
  if (pieces.length === 0) return null

  const diag = pieces.find(w => w.diagonal)
  if (diag) {
    const ax = diag.ax as number, ay = diag.ay as number, bx = diag.bx as number, by = diag.by as number
    const len = Math.hypot(bx - ax, by - ay)
    return {
      horizontal: false, diagonal: true, perp: 0, thickness: diag.thickness,
      loBase: 0, hiBase: len, startId: diag.id, endId: diag.id, startLen: len, endLen: len,
      ax, ay, bx, by,
    }
  }

  const hz = pieces[0].horizontal
  const lo = (w: Wall) => (hz ? w.x0 : w.y0)
  const hi = (w: Wall) => (hz ? w.x1 : w.y1)
  const first = pieces.reduce((a, w) => (lo(w) < lo(a) ? w : a))
  const last = pieces.reduce((a, w) => (hi(w) > hi(a) ? w : a))
  const perp = hz ? (first.y0 + first.y1) / 2 : (first.x0 + first.x1) / 2
  return {
    horizontal: hz, diagonal: false, perp, thickness: first.thickness,
    loBase: lo(first), hiBase: hi(last),
    startId: first.id, endId: last.id,
    startLen: hi(first) - lo(first), endLen: hi(last) - lo(last),
  }
}

// Keep an extension sensible: a wall can't be shortened past its end piece or below a minimum length.
export function clampExt(info: AxisInfo, ext: Ext): Ext {
  const start = Math.min(MAX_EXTEND, Math.max(-(info.startLen - MIN_PIECE), ext.start))
  let end = Math.min(MAX_EXTEND, Math.max(-(info.endLen - MIN_PIECE), ext.end))
  const len = info.hiBase - info.loBase + start + end
  if (len < MIN_LENGTH) end += MIN_LENGTH - len
  return { start, end }
}

// The walls with every extension applied (what is drawn and measured).
export function applyExtensions(walls: Wall[], groups: WallGroups, ext: Record<number, Ext>): Wall[] {
  const keys = Object.keys(ext)
  if (keys.length === 0) return walls

  const adj = new Map<number, { s: number; e: number }>()
  const put = (id: number, s: number, e: number) => {
    const a = adj.get(id) ?? { s: 0, e: 0 }
    adj.set(id, { s: a.s + s, e: a.e + e })
  }
  for (const k of keys) {
    const info = groupAxis(walls, groups, Number(k))
    if (!info) continue
    const v = ext[Number(k)]
    if (info.startId === info.endId) put(info.startId, v.start, v.end)
    else {
      put(info.startId, v.start, 0)
      put(info.endId, 0, v.end)
    }
  }

  return walls.map(w => {
    const a = adj.get(w.id)
    if (!a) return w
    if (w.diagonal) {
      const ax = w.ax as number, ay = w.ay as number, bx = w.bx as number, by = w.by as number
      const len = Math.hypot(bx - ax, by - ay) || 1
      const dx = (bx - ax) / len, dy = (by - ay) / len
      const nax = ax - dx * a.s, nay = ay - dy * a.s
      const nbx = bx + dx * a.e, nby = by + dy * a.e
      return {
        ...w, ax: nax, ay: nay, bx: nbx, by: nby, length: len + a.s + a.e,
        x0: Math.min(nax, nbx), x1: Math.max(nax, nbx), y0: Math.min(nay, nby), y1: Math.max(nay, nby),
      }
    }
    if (w.horizontal) {
      const x0 = w.x0 - a.s, x1 = w.x1 + a.e
      return { ...w, x0, x1, length: x1 - x0 }
    }
    const y0 = w.y0 - a.s, y1 = w.y1 + a.e
    return { ...w, y0, y1, length: y1 - y0 }
  })
}

export interface SnapResult { value: number; targetId: number | null; kind: 'touch' | 'align' | null }

// Snap the coordinate of a wall end (along its axis) to a nearby wall:
//  - 'touch': the face of a crossing wall, so the extended wall butts against it
//  - 'align': the end of a parallel wall, so the two finish in line
export function snapCoordinate(o: {
  walls: Wall[]
  skip: number[]
  horizontal: boolean
  perp: number
  value: number
  threshold: number
}): SnapResult {
  const skip = new Set(o.skip)
  const tol = 0.05
  let best: SnapResult = { value: o.value, targetId: null, kind: null }
  let bestDist = o.threshold

  const consider = (c: number, id: number, kind: 'touch' | 'align') => {
    const d = Math.abs(c - o.value)
    // a touch within 2 cm of an align wins, because a clean corner matters more
    if (d < bestDist - (kind === 'touch' ? 0.02 : 0) || (best.kind === 'align' && kind === 'touch' && d <= bestDist + 0.02)) {
      best = { value: c, targetId: id, kind }
      bestDist = d
    }
  }

  for (const w of o.walls) {
    if (skip.has(w.id) || w.diagonal) continue
    if (w.horizontal === o.horizontal) {
      consider(o.horizontal ? w.x0 : w.y0, w.id, 'align')
      consider(o.horizontal ? w.x1 : w.y1, w.id, 'align')
    } else {
      // a crossing wall only counts if it actually reaches this wall's centre line
      const lo = o.horizontal ? w.y0 : w.x0
      const hi = o.horizontal ? w.y1 : w.x1
      if (o.perp >= lo - tol && o.perp <= hi + tol) {
        consider(o.horizontal ? w.x0 : w.y0, w.id, 'touch')
        consider(o.horizontal ? w.x1 : w.y1, w.id, 'touch')
      }
    }
  }
  return best
}