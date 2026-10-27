// Mark the floor plans by hand: draw a rectangle round each plan on the detection overlay, ground
// floor first. The backend then gives every wall, door, window, stair and railing to its plan and
// works out how to lay the plans over each other (model.floors).
import { useRef, useState } from 'react'
import type { PlanModel } from './types'

/* eslint-disable @typescript-eslint/no-explicit-any */
interface Box { x0: number; y0: number; x1: number; y1: number; label: string }
interface Props {
  model: PlanModel
  /** ask the backend: boxes in plan metres (ground first) -> floors list, or null; no boxes = automatic */
  detect: (boxes: number[][], labels: string[], count: number) => Promise<any[] | null>
  onApply: (floors: any[] | null) => void
  onClose: () => void
}
const NAMES = ['Ground floor', 'First floor', 'Second floor', 'Third floor', 'Fourth floor']

export default function FloorMarker({ model, detect, onApply, onClose }: Props) {
  const m: any = model
  const ppm: number = m.px_per_m
  const svg = useRef<SVGSVGElement>(null)
  const [size, setSize] = useState({ w: 1000, h: 800 })
  const [boxes, setBoxes] = useState<Box[]>(() =>
    ((m.floors ?? []) as any[]).map((f, i) => ({ ...f.bbox, label: f.label ?? NAMES[i] ?? `Floor ${i + 1}` })))
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')
  const [count, setCount] = useState(2)

  const at = (e: React.PointerEvent): [number, number] => {
    const r = svg.current!.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * size.w / ppm, ((e.clientY - r.top) / r.height) * size.h / ppm]
  }
  const down = (e: React.PointerEvent) => { const p = at(e); setDrag({ a: p, b: p }); (e.target as Element).setPointerCapture?.(e.pointerId) }
  const move = (e: React.PointerEvent) => { if (drag) setDrag({ ...drag, b: at(e) }) }
  const up = () => {
    if (!drag) return
    const x0 = Math.min(drag.a[0], drag.b[0]), x1 = Math.max(drag.a[0], drag.b[0])
    const y0 = Math.min(drag.a[1], drag.b[1]), y1 = Math.max(drag.a[1], drag.b[1])
    setDrag(null)
    if (x1 - x0 < 2 || y1 - y0 < 2) return
    setBoxes(b => [...b, { x0, y0, x1, y1, label: NAMES[b.length] ?? `Floor ${b.length + 1}` }])
  }
  const swap = (i: number, j: number) => setBoxes(b => { const c = [...b]; [c[i], c[j]] = [c[j], c[i]]; return c })

  const run = async (boxesIn: Box[], n: number) => {
    setBusy(true); setMsg('')
    try {
      const f = await detect(boxesIn.map(b => [b.x0, b.y0, b.x1, b.y1]), boxesIn.map(b => b.label), n)
      if (!f) { setMsg(boxesIn.length ? 'Could not build floors from these boxes: each must contain walls.' : 'No separate plans found. Draw a box round each plan instead.'); return }
      onApply(f)
    } catch (e) { setMsg('Failed: ' + (e as Error).message) } finally { setBusy(false) }
  }

  const px = (v: number) => v * ppm
  const colours = ['#1e90ff', '#ff8c00', '#2ca02c', '#c2185b', '#7b1fa2']
  const fs = Math.max(14, size.w / 55)
  const rect = (b: Box, i: number, live = false) => (
    <g key={i}>
      <rect x={px(b.x0)} y={px(b.y0)} width={px(b.x1 - b.x0)} height={px(b.y1 - b.y0)}
            fill={colours[i % 5]} fillOpacity={live ? 0.15 : 0.18} stroke={colours[i % 5]} strokeWidth={Math.max(2, size.w / 400)} />
      {!live && <text x={px(b.x0) + 8} y={px(b.y0) + fs + 4} fontSize={fs} fontWeight={700} fill={colours[i % 5]}>{i + 1}. {b.label}</text>}
    </g>
  )

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 60, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ background: '#fff', color: '#111', borderRadius: 10, width: 'min(1200px, 96vw)', height: '92vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', padding: '8px 12px', borderBottom: '1px solid #ddd' }}>
          <strong>Mark the floor plans</strong>
          <span style={{ color: '#555', fontSize: 13 }}>Drag a box round each plan, ground floor first.</span>
          <span style={{ flex: 1 }} />
          <label style={{ fontSize: 13 }}>Auto: plans on sheet
            <select value={count} onChange={e => setCount(Number(e.target.value))} style={{ marginLeft: 4 }}>
              <option value={0}>find automatically</option>
              <option value={2}>2</option><option value={3}>3</option><option value={4}>4</option>
            </select></label>
          <button disabled={busy} onClick={() => run([], count)}>Auto-detect</button>
          <button disabled={busy} onClick={() => onApply(null)}>Single floor (no stacking)</button>
          <button onClick={onClose}>Cancel</button>
          <button disabled={busy || boxes.length < 2} onClick={() => run(boxes, 0)} style={{ fontWeight: 700 }}>
            {busy ? 'Working…' : `Use these ${boxes.length} floors`}</button>
        </div>
        {msg && <div style={{ padding: '4px 12px', color: '#b00', fontSize: 13 }}>{msg}</div>}
        <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
          <div style={{ width: 230, borderRight: '1px solid #ddd', overflow: 'auto', padding: 8, fontSize: 13 }}>
            {boxes.length === 0 && <p style={{ color: '#666' }}>No floors marked yet. Draw one box per plan on the image.</p>}
            {boxes.map((b, i) => (
              <div key={i} style={{ borderLeft: `4px solid ${colours[i % 5]}`, padding: '4px 6px', marginBottom: 6, background: '#f6f6f6' }}>
                <input value={b.label} onChange={e => setBoxes(bs => bs.map((x, k) => k === i ? { ...x, label: e.target.value } : x))} style={{ width: '100%' }} />
                <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                  <button disabled={i === 0} onClick={() => swap(i, i - 1)} title="Lower (towards ground)">↑</button>
                  <button disabled={i === boxes.length - 1} onClick={() => swap(i, i + 1)} title="Higher">↓</button>
                  <button onClick={() => setBoxes(bs => bs.filter((_, k) => k !== i))}>Delete</button>
                </div>
              </div>
            ))}
            {boxes.length > 0 && <button onClick={() => setBoxes([])}>Clear all</button>}
            <p style={{ color: '#666', marginTop: 10 }}>The first in the list is the ground floor; the next ones are built on top of it in this order.</p>
          </div>
          <div style={{ flex: 1, overflow: 'auto', background: '#eee' }}>
            <div style={{ position: 'relative', width: '100%' }}>
              <img src={m.overlay} alt="plan" draggable={false} style={{ display: 'block', width: '100%' }}
                   onLoad={e => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
              <svg ref={svg} viewBox={`0 0 ${size.w} ${size.h}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', cursor: 'crosshair', touchAction: 'none' }}
                   onPointerDown={down} onPointerMove={move} onPointerUp={up}>
                {boxes.map((b, i) => rect(b, i))}
                {drag && rect({ x0: Math.min(drag.a[0], drag.b[0]), y0: Math.min(drag.a[1], drag.b[1]), x1: Math.max(drag.a[0], drag.b[0]), y1: Math.max(drag.a[1], drag.b[1]), label: '' }, boxes.length, true)}
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}