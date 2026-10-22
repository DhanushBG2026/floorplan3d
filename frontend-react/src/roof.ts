// Roof geometry as plain triangle lists (x, y, z per vertex). Plan coordinates: x right, z = plan y,
// y up. The walls' outline is (x0, z0)-(x1, z1) and their top is at eaveY.
export type RoofType = 'none' | 'gable' | 'hip' | 'skillion' | 'flat'

export interface RoofInput {
  type: RoofType
  x0: number
  z0: number
  x1: number
  z1: number
  eaveY: number
  ridgeY: number
  overhang: number
}

export interface RoofTriangles {
  roof: number[]
  gables: number[] // vertical infill triangles that close the walls under the roof
}

type V = [number, number, number]

function quad(a: V, b: V, c: V, d: V): number[] {
  return [...a, ...b, ...c, ...a, ...c, ...d]
}
function tri(a: V, b: V, c: V): number[] {
  return [...a, ...b, ...c]
}

// Build in (u, y, v) space where u runs along the ridge and v across it, then map to (x, z).
export function roofTriangles(p: RoofInput): RoofTriangles {
  const empty: RoofTriangles = { roof: [], gables: [] }
  if (p.type === 'none') return empty

  const w = p.x1 - p.x0
  const d = p.z1 - p.z0
  if (w <= 0 || d <= 0) return empty

  if (p.type === 'flat') {
    const y = p.eaveY + 0.02
    const o = p.overhang
    return {
      roof: quad([p.x0 - o, y, p.z0 - o], [p.x1 + o, y, p.z0 - o], [p.x1 + o, y, p.z1 + o], [p.x0 - o, y, p.z1 + o]),
      gables: [],
    }
  }

  const alongX = w >= d // ridge runs along the longer side
  const u0 = alongX ? p.x0 : p.z0
  const u1 = alongX ? p.x1 : p.z1
  const v0 = alongX ? p.z0 : p.x0
  const v1 = alongX ? p.z1 : p.x1
  const map = (u: number, y: number, v: number): V => (alongX ? [u, y, v] : [v, y, u])

  const o = p.overhang
  const eave = p.eaveY
  const ridge = Math.max(p.ridgeY, eave + 0.05)
  const U0 = u0 - o, U1 = u1 + o, V0 = v0 - o, V1 = v1 + o
  const vc = (v0 + v1) / 2
  const run = (v1 - v0) / 2
  const slope = (ridge - eave) / run
  const yE = eave - slope * o // height of the eave edge once the roof overhangs the wall

  const roof: number[] = []
  const gables: number[] = []

  if (p.type === 'gable') {
    roof.push(...quad(map(U0, yE, V0), map(U1, yE, V0), map(U1, ridge, vc), map(U0, ridge, vc)))
    roof.push(...quad(map(U1, yE, V1), map(U0, yE, V1), map(U0, ridge, vc), map(U1, ridge, vc)))
    gables.push(...tri(map(u0, eave, v0), map(u0, eave, v1), map(u0, ridge, vc)))
    gables.push(...tri(map(u1, eave, v1), map(u1, eave, v0), map(u1, ridge, vc)))
  } else if (p.type === 'hip') {
    const half = Math.max(0, ((u1 - u0) - (v1 - v0)) / 2) // half the ridge length
    const uc = (u0 + u1) / 2
    const r0 = uc - half
    const r1 = uc + half
    roof.push(...quad(map(U0, yE, V0), map(U1, yE, V0), map(r1, ridge, vc), map(r0, ridge, vc)))
    roof.push(...quad(map(U1, yE, V1), map(U0, yE, V1), map(r0, ridge, vc), map(r1, ridge, vc)))
    roof.push(...tri(map(U0, yE, V1), map(U0, yE, V0), map(r0, ridge, vc)))
    roof.push(...tri(map(U1, yE, V0), map(U1, yE, V1), map(r1, ridge, vc)))
  } else {
    // skillion: high side at v0, falling towards v1
    const full = v1 - v0
    const s = (ridge - eave) / full
    const yHigh = ridge + s * o
    const yLow = eave - s * o
    roof.push(...quad(map(U0, yHigh, V0), map(U1, yHigh, V0), map(U1, yLow, V1), map(U0, yLow, V1)))
    // close the walls under the sloping roof
    gables.push(...quad(map(u0, eave, v0), map(u1, eave, v0), map(u1, ridge, v0), map(u0, ridge, v0)))
    gables.push(...tri(map(u0, eave, v0), map(u0, eave, v1), map(u0, ridge, v0)))
    gables.push(...tri(map(u1, eave, v1), map(u1, eave, v0), map(u1, ridge, v0)))
  }
  return { roof, gables }
}

// Texture coordinates for a flat-shaded triangle list (9 numbers per triangle): every triangle is
// projected onto its own plane, in metres divided by `tile`, so a texture keeps its real-world size.
export function planarUVs(pos: number[], tile: number, swap = false): number[] {
  const uv: number[] = []
  const sub = (a: number[], b: number[]) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
  const cross = (a: number[], b: number[]) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
  const len = (a: number[]) => Math.hypot(a[0], a[1], a[2])
  const unit = (a: number[], fallback: number[]) => {
    const l = len(a)
    return l < 1e-9 ? fallback : [a[0] / l, a[1] / l, a[2] / l]
  }
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

  for (let i = 0; i + 8 < pos.length; i += 9) {
    const p = [0, 1, 2].map(k => [pos[i + k * 3], pos[i + k * 3 + 1], pos[i + k * 3 + 2]])
    const n = unit(cross(sub(p[1], p[0]), sub(p[2], p[0])), [0, 1, 0])
    const u = unit(cross(n, [0, 1, 0]), [1, 0, 0]) // horizontal direction inside the plane
    const v = unit(cross(n, u), [0, 0, 1])
    for (const q of p) {
      const a = dot(q, u) / tile
      const b = dot(q, v) / tile
      uv.push(swap ? b : a, swap ? a : b)
    }
  }
  return uv
}