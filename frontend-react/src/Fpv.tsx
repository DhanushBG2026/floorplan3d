import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useFrame, useThree } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import type { Opening, Wall } from './types'

// door id -> how far its leaf is told to swing (radians, signed). 0 = closed.
// The walker writes it, DoorParts in Viewer.tsx animates towards it.
export const doorTargets = new Map<number, number>()
const DOOR_SWING = 1.6          // ~92 degrees
const EYE = 1.6                 // eye height above the floor (m)
const RADIUS = 0.25             // body radius (m)
const DEFAULT_FOV = 70

interface Seg { ax: number; ay: number; bx: number; by: number; r: number }

// Everything the walker bumps into, as thick line segments in plan coordinates.
// Door openings are left open (the walker can pass); windows stay solid.
function buildColliders(walls: Wall[], openings: Opening[], thick: (w: Wall) => number): Seg[] {
  const out: Seg[] = []
  for (const w of walls) {
    const t = thick(w) || 0.12
    if (w.arc) {
      const a = w.arc, n = Math.max(6, Math.ceil(Math.abs(a.a1 - a.a0) / (Math.PI / 18)))
      for (let i = 0; i < n; i++) {
        const t0 = a.a0 + ((a.a1 - a.a0) * i) / n, t1 = a.a0 + ((a.a1 - a.a0) * (i + 1)) / n
        out.push({
          ax: a.cx + a.r * Math.cos(t0), ay: a.cy + a.r * Math.sin(t0),
          bx: a.cx + a.r * Math.cos(t1), by: a.cy + a.r * Math.sin(t1), r: t / 2,
        })
      }
      continue
    }
    if (w.diagonal) {
      out.push({ ax: w.ax as number, ay: w.ay as number, bx: w.bx as number, by: w.by as number, r: t / 2 })
      continue
    }
    const h = !!w.horizontal
    const start = h ? w.x0 : w.y0, end = h ? w.x1 : w.y1
    const centre = h ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
    const cuts = openings
      .filter(o => o.type === 'door' && o.horizontal === h)
      .filter(o => Math.abs(o.centre - centre) <= Math.max(t, o.thickness || 0, 0.12))
      .map(o => [Math.max(start, o.start), Math.min(end, o.end)] as [number, number])
      .filter(([a, b]) => b - a > 0.015)
      .sort((a, b) => a[0] - b[0])
    let cursor = start
    const push = (a: number, b: number) => {
      if (b - a > 0.015) out.push(h ? { ax: a, ay: centre, bx: b, by: centre, r: t / 2 }
                                    : { ax: centre, ay: a, bx: centre, by: b, r: t / 2 })
    }
    for (const [a, b] of cuts) { push(cursor, a); cursor = Math.max(cursor, b) }
    push(cursor, end)
  }
  return out
}

function pushOut(p: { x: number; z: number }, segs: Seg[]) {
  for (let pass = 0; pass < 3; pass++) {
    for (const s of segs) {
      const dx = s.bx - s.ax, dy = s.by - s.ay
      const l2 = dx * dx + dy * dy
      let u = l2 > 0 ? ((p.x - s.ax) * dx + (p.z - s.ay) * dy) / l2 : 0
      u = Math.max(0, Math.min(1, u))
      const qx = s.ax + u * dx, qy = s.ay + u * dy
      let ex = p.x - qx, ey = p.z - qy
      const d = Math.hypot(ex, ey), min = RADIUS + s.r
      if (d >= min) continue
      if (d < 1e-6) { const n = Math.hypot(dx, dy) || 1; ex = -dy / n; ey = dx / n } else { ex /= d; ey /= d }
      p.x = qx + ex * min
      p.z = qy + ey * min
    }
  }
}

interface FpvProps {
  walls: Wall[]
  openings: Opening[]
  offset: [number, number, number]       // the level's offset in the scene
  floorY: number                          // world height of the floor (stumps)
  bounds: { x0: number; y0: number; x1: number; y1: number }
  zoneOf?: Record<number, 'ext' | 'int'>
  thick?: (w: Wall) => number
  onExit: () => void
}

export function FpvController({ walls, openings, offset, floorY, bounds, zoneOf = {}, thick, onExit }: FpvProps) {
  const camera = useThree(s => s.camera) as THREE.PerspectiveCamera
  const dom = useThree(s => s.gl.domElement)
  const exitRef = useRef(onExit); exitRef.current = onExit

  const segs = useMemo(
    () => buildColliders(walls, openings, thick ?? (w => Number(w.thickness) || 0.12)),
    [walls, openings, thick],
  )
  const doors = useMemo(() => openings.filter(o => o.type === 'door'), [openings])

  const st = useRef({ x: 0, z: 0, yaw: 0, pitch: 0, fov: DEFAULT_FOV, keys: new Set<string>(), locked: false, drag: false })

  // start: just outside a door, looking in (inside it when the house stands on stumps).
  // "In" is the side of the door where the nearest wall is, i.e. where the house is.
  useEffect(() => {
    const s = st.current
    const hit = (px: number, pz: number, dx: number, dz: number) => {
      let best = Infinity
      for (const g of segs) {
        const ex = g.bx - g.ax, ey = g.by - g.ay
        const den = dx * ey - dz * ex
        if (Math.abs(den) < 1e-9) continue
        const t = ((g.ax - px) * ey - (g.ay - pz) * ex) / den
        const u = ((g.ax - px) * dz - (g.ay - pz) * dx) / den
        if (t > 0.3 && u >= 0 && u <= 1 && t < best) best = t
      }
      return best
    }
    const cands = doors.map(o => {
      const mid = (o.start + o.end) / 2
      const px = o.horizontal ? mid : o.centre, pz = o.horizontal ? o.centre : mid
      const ax = o.horizontal ? 0 : 1, az = o.horizontal ? 1 : 0       // the door's perpendicular
      const h1 = hit(px, pz, ax, az), h2 = hit(px, pz, -ax, -az)
      const sign = h1 <= h2 ? 1 : -1                                  // towards the nearer wall = inside
      return { o, px, pz, dx: ax * sign, dz: az * sign, inside: Math.min(h1, h2), ext: zoneOf[o.id] === 'ext' ? 0 : 1 }
    }).filter(c => c.inside < 40)
    cands.sort((p, q) => p.ext - q.ext || q.inside - p.inside)         // exterior first, then the roomier side
    const c = cands[0]
    if (c) {
      const dist = floorY > 0.05 ? 1.3 : -2.2
      s.x = c.px + c.dx * dist; s.z = c.pz + c.dz * dist
      s.yaw = Math.atan2(-c.dx, -c.dz)
    } else {
      s.x = (bounds.x0 + bounds.x1) / 2; s.z = (bounds.y0 + bounds.y1) / 2; s.yaw = 0
    }
    pushOut(s, segs)
    s.pitch = 0; s.fov = DEFAULT_FOV
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // enter / leave: remember the orbit camera, take over, hand it back
  useEffect(() => {
    const saved = { pos: camera.position.clone(), fov: camera.fov }
    camera.rotation.order = 'YXZ'
    const s = st.current

    const lock = () => { try { (dom.requestPointerLock() as unknown as Promise<void> | undefined)?.catch?.(() => {}) } catch { /* not allowed */ } }
    lock()

    const typing = (t: EventTarget | null) => {
      const el = t as HTMLElement | null
      return !!el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)
    }
    const used = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ShiftLeft', 'ShiftRight', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'])
    const down = (e: KeyboardEvent) => {
      if (typing(e.target)) return
      if (e.code === 'Escape') { exitRef.current(); return }
      if (used.has(e.code)) { s.keys.add(e.code); e.preventDefault() }
    }
    const up = (e: KeyboardEvent) => { s.keys.delete(e.code) }
    const blur = () => s.keys.clear()
    const move = (e: MouseEvent) => {
      if (!s.locked && !s.drag) return
      s.yaw -= e.movementX * 0.0022
      s.pitch = Math.max(-1.45, Math.min(1.45, s.pitch - e.movementY * 0.0022))
    }
    const plc = () => {
      const on = document.pointerLockElement === dom
      if (on) s.locked = true
      else if (s.locked) { s.locked = false; exitRef.current() }   // Esc while the mouse is captured
    }
    const pdown = () => { if (!s.locked) { s.drag = true; lock() } }
    const pup = () => { s.drag = false }
    const wheel = (e: WheelEvent) => { s.fov = Math.max(20, Math.min(100, s.fov + e.deltaY * 0.03)); e.preventDefault() }

    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', blur)
    document.addEventListener('mousemove', move)
    document.addEventListener('pointerlockchange', plc)
    dom.addEventListener('pointerdown', pdown)
    window.addEventListener('pointerup', pup)
    dom.addEventListener('wheel', wheel, { passive: false })
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', blur)
      document.removeEventListener('mousemove', move)
      document.removeEventListener('pointerlockchange', plc)
      dom.removeEventListener('pointerdown', pdown)
      window.removeEventListener('pointerup', pup)
      dom.removeEventListener('wheel', wheel)
      s.locked = false
      if (document.pointerLockElement === dom) document.exitPointerLock()
      doorTargets.clear()
      camera.position.copy(saved.pos)
      camera.fov = saved.fov
      camera.updateProjectionMatrix()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05)
    const s = st.current
    const k = s.keys
    const f = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0)
    const r = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0)
    if (k.has('ArrowLeft')) s.yaw += 1.8 * dt
    if (k.has('ArrowRight')) s.yaw -= 1.8 * dt
    if (f || r) {
      const speed = (k.has('ShiftLeft') || k.has('ShiftRight') ? 4.2 : 1.9) * dt
      const len = Math.hypot(f, r)
      const sin = Math.sin(s.yaw), cos = Math.cos(s.yaw)
      s.x += ((-sin * f + cos * r) / len) * speed
      s.z += ((-cos * f - sin * r) / len) * speed
      pushOut(s, segs)
    }
    if (k.has('KeyQ')) s.fov = Math.max(20, s.fov - 45 * dt)     // zoom in
    if (k.has('KeyE')) s.fov = Math.min(100, s.fov + 45 * dt)    // zoom out

    // doors swing open away from you when you come close, and shut behind you
    for (const o of doors) {
      const along = o.horizontal ? s.x : s.z
      const perp = (o.horizontal ? s.z : s.x) - o.centre
      const cur = doorTargets.get(o.id) ?? 0
      const m = cur !== 0 ? 0.5 : 0
      const near = along > o.start - 0.8 - m && along < o.end + 0.8 + m && Math.abs(perp) < 1.3 + m
      if (near && cur === 0) {
        const away = perp > 0 ? -1 : 1                      // +1 = leaf swings to the +perp side
        doorTargets.set(o.id, (o.horizontal ? -away : away) * DOOR_SWING)
      } else if (!near && cur !== 0) doorTargets.set(o.id, 0)
    }

    camera.position.set(s.x + offset[0], floorY + offset[1] + EYE, s.z + offset[2])
    camera.rotation.set(s.pitch, s.yaw, 0, 'YXZ')
    if (Math.abs(camera.fov - s.fov) > 0.01) { camera.fov = s.fov; camera.updateProjectionMatrix() }
  })

  return (
    <Html fullscreen zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', fontFamily: 'system-ui, sans-serif' }}>
        <div style={{ position: 'absolute', left: '50%', top: '50%', width: 6, height: 6, margin: -3, borderRadius: 3, background: 'rgba(255,255,255,.85)', boxShadow: '0 0 0 1px rgba(0,0,0,.5)' }} />
        <div style={{ position: 'absolute', left: '50%', bottom: 18, transform: 'translateX(-50%)', background: 'rgba(0,0,0,.6)', color: '#fff', padding: '6px 12px', borderRadius: 8, fontSize: 12, whiteSpace: 'nowrap' }}>
          W A S D move · Mouse look · Q / E zoom · Shift run · Esc exit
        </div>
      </div>
    </Html>
  )
}