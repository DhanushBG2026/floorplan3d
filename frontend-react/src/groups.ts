import type { PlanModel } from './types'

export interface WallGroups {
  groupOf: Record<number, number>       // wall piece id -> wall (group) id
  members: Record<number, number[]>     // wall (group) id -> all piece ids
  openingGroup: Record<number, number>  // opening id -> wall (group) id
  length: Record<number, number>        // wall (group) id -> full length incl. openings (m)
}

// Wall pieces separated by a door/window are one wall. The group id is the smallest piece id.
export function buildGroups(model: PlanModel, tol = 0.1): WallGroups {
  const parent: Record<number, number> = {}
  const find = (a: number): number => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]]
      a = parent[a]
    }
    return a
  }
  const union = (a: number, b: number) => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb)
  }
  model.walls.forEach(w => { parent[w.id] = w.id })

  const attach: Record<number, number> = {}
  for (const o of model.openings) {
    let before: number | null = null
    let after: number | null = null
    for (const w of model.walls) {
      
      if (w.diagonal) continue
      if (w.horizontal !== o.horizontal) continue
      const centre = o.horizontal ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
      if (Math.abs(centre - o.centre) > Math.max(tol, o.thickness / 2)) continue
      const a0 = o.horizontal ? w.x0 : w.y0
      const a1 = o.horizontal ? w.x1 : w.y1
      if (Math.abs(a1 - o.start) <= tol) before = w.id
      if (Math.abs(a0 - o.end) <= tol) after = w.id
    }
    if (before !== null && after !== null) union(before, after)
    const any = before ?? after
    if (any !== null) attach[o.id] = any
  }

  const groupOf: Record<number, number> = {}
  const members: Record<number, number[]> = {}
  const openingGroup: Record<number, number> = {}
  const lo: Record<number, number> = {}
  const hi: Record<number, number> = {}
  const extend = (g: number, a: number, b: number) => {
    lo[g] = Math.min(lo[g] ?? a, a)
    hi[g] = Math.max(hi[g] ?? b, b)
  }

  for (const w of model.walls) {
    const g = find(w.id)
    groupOf[w.id] = g
    ;(members[g] ??= []).push(w.id)
    if (w.diagonal) {
    // a diagonal wall is its own group; remember its real length
    lo[g] = 0
    hi[g] = w.length
    } else {
    extend(g, w.horizontal ? w.x0 : w.y0, w.horizontal ? w.x1 : w.y1)
    }
  }
  for (const o of model.openings) {
    if (attach[o.id] === undefined) continue
    const g = find(attach[o.id])
    openingGroup[o.id] = g
    extend(g, o.start, o.end)
  }
  const length: Record<number, number> = {}
  for (const g of Object.keys(members)) length[+g] = hi[+g] - lo[+g]

  return { groupOf, members, openingGroup, length }
}