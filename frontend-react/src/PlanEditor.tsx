// Draw extra walls, doors and windows on top of the plan image. Everything is in plan metres
// (1 px of the image = 1 / px_per_m), so what you add lands exactly where it is drawn in the 3D.
import { useMemo, useRef, useState } from 'react'
import type { PlanModel } from './types'

type Tool = 'wall' | 'door' | 'window' | 'railing' | 'deck' | 'erase'
export interface Extras { railings: any[]; decks: any[] }
/* eslint-disable @typescript-eslint/no-explicit-any */
type W = any
type O = any
const FT = 0.3048

interface Props {
  image: string
  model: PlanModel
  onApply: (walls: W[], openings: O[], extras: Extras) => void
  onClose: () => void
}

const dist = (a: number, b: number, c: number, d: number) => Math.hypot(a - c, b - d)

export default function PlanEditor({ image, model, onApply, onClose }: Props) {
  const m: any = model
  const ppm: number = m.px_per_m
  const imgRef = useRef<HTMLImageElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [size, setSize] = useState({ w: 1000, h: 800 })
  const [tool, setTool] = useState<Tool>('wall')
  const [zoom, setZoom] = useState(1)
  const [wallFt, setWallFt] = useState(() => {
    const th = (m.walls as W[]).filter(w => !w.diagonal).map(w => w.thickness).sort((a: number, b: number) => a - b)
    return Math.round(((th[Math.floor(th.length / 2)] || 0.15) / FT) * 12) / 12
  })
  const [doorFt, setDoorFt] = useState(3.5)
  const [winFt, setWinFt] = useState(4)
  const [newWalls, setNewWalls] = useState<W[]>([])
  const [newOpenings, setNewOpenings] = useState<O[]>([])
  const [newRailings, setNewRailings] = useState<any[]>([])
  const [newDecks, setNewDecks] = useState<any[]>([])
  const [deckRoof, setDeckRoof] = useState(false)
  const [drag, setDrag] = useState<{ a: [number, number]; b: [number, number] } | null>(null)
  const [hover, setHover] = useState<[number, number] | null>(null)
  const [hist, setHist] = useState<{ w: W[]; o: O[]; r: any[]; d: any[] }[]>([])

  const baseId = useMemo(() => {
    const ids = [...m.walls, ...m.openings].map((x: any) => x.id as number)
    return { wall: Math.max(-1, ...m.walls.map((w: W) => w.id)) + 1, op: Math.max(-1, ...m.openings.map((o: O) => o.id)) + 1, max: Math.max(-1, ...ids),
             rail: Math.max(-1, ...(m.railings ?? []).map((r: any) => r.id)) + 1, deck: Math.max(-1, ...(m.decks ?? []).map((d: any) => d.id)) + 1 }
  }, [m])
  const allWalls: W[] = useMemo(() => [...m.walls, ...newWalls], [m.walls, newWalls])

  // points a new wall end may snap to: corners of every straight wall's box and ends of diagonals
  const snapPts = useMemo(() => {
    const p: [number, number][] = []
    for (const w of allWalls) {
      if (w.diagonal) { p.push([w.ax, w.ay], [w.bx, w.by]); continue }
      const cy = (w.y0 + w.y1) / 2, cx = (w.x0 + w.x1) / 2
      if (w.horizontal) p.push([w.x0, cy], [w.x1, cy]); else p.push([cx, w.y0], [cx, w.y1])
    }
    return p
  }, [allWalls])

  const snapshot = () => setHist(h => [...h, { w: newWalls, o: newOpenings, r: newRailings, d: newDecks }])
  const undo = () => setHist(h => {
    if (!h.length) return h
    const last = h[h.length - 1]
    setNewWalls(last.w); setNewOpenings(last.o); setNewRailings(last.r); setNewDecks(last.d)
    return h.slice(0, -1)
  })

  const toPlan = (e: React.PointerEvent): [number, number] => {
    const svg = svgRef.current!
    const r = svg.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * size.w / ppm, ((e.clientY - r.top) / r.height) * size.h / ppm]
  }
  const snapPoint = (p: [number, number], from?: [number, number], free = false): [number, number] => {
    const tol = 0.25
    let best: [number, number] | null = null, bd = tol
    for (const q of snapPts) { const d = dist(p[0], p[1], q[0], q[1]); if (d < bd) { bd = d; best = q } }
    let [x, y] = best ?? p
    if (from && !free) {                                   // keep walls straight unless Alt is held
      const dx = x - from[0], dy = y - from[1]
      const ang = Math.abs(Math.atan2(dy, dx)) * 180 / Math.PI
      if (ang < 8 || ang > 172) y = from[1]
      else if (Math.abs(ang - 90) < 8) x = from[0]
    }
    return [x, y]
  }

  // the straight wall under a point, with where along it the point falls
  const wallAt = (p: [number, number]) => {
    let best: { w: W; along: number; cross: number; d: number } | null = null
    for (const w of allWalls) {
      if (w.diagonal) continue
      const along = w.horizontal ? p[0] : p[1]
      const a0 = w.horizontal ? w.x0 : w.y0, a1 = w.horizontal ? w.x1 : w.y1
      const cross = w.horizontal ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
      const d = Math.abs((w.horizontal ? p[1] : p[0]) - cross)
      if (d <= Math.max(w.thickness / 2, 0.15) + 0.1 && along >= a0 - 0.05 && along <= a1 + 0.05 && (!best || d < best.d))
        best = { w, along, cross, d }
    }
    return best
  }
  const openingFor = (tool: 'door' | 'window', p: [number, number]): O | null => {
    const hit = wallAt(p)
    if (!hit) return null
    const w = hit.w
    const width = (tool === 'door' ? doorFt : winFt) * FT
    const a0 = w.horizontal ? w.x0 : w.y0, a1 = w.horizontal ? w.x1 : w.y1
    if (a1 - a0 < width) return null
    const c = Math.min(Math.max(hit.along, a0 + width / 2), a1 - width / 2)
    return { type: tool, horizontal: !!w.horizontal, start: c - width / 2, end: c + width / 2, centre: hit.cross,
             thickness: w.thickness, width, ink_density: 1, added: true }
  }

  const makeWall = (a: [number, number], b: [number, number], t: number): W | null => {
    const len = dist(a[0], a[1], b[0], b[1])
    if (len < 0.3) return null
    const id = baseId.wall + newWalls.length
    const straightH = Math.abs(a[1] - b[1]) < 1e-6, straightV = Math.abs(a[0] - b[0]) < 1e-6
    if (straightH || straightV) {
      const h = straightH
      const lo = h ? Math.min(a[0], b[0]) : Math.min(a[1], b[1]), hi = h ? Math.max(a[0], b[0]) : Math.max(a[1], b[1])
      const c = h ? a[1] : a[0]
      const e = t / 2                                     // ends reach the far face of the wall they meet
      return h
        ? { id, x0: lo - e, y0: c - t / 2, x1: hi + e, y1: c + t / 2, horizontal: true, length: hi - lo + t, thickness: t, added: true }
        : { id, x0: c - t / 2, y0: lo - e, x1: c + t / 2, y1: hi + e, horizontal: false, length: hi - lo + t, thickness: t, added: true }
    }
    const th = Math.atan2(b[1] - a[1], b[0] - a[0])
    const dx = Math.cos(th) * len / 2, dy = Math.sin(th) * len / 2
    const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2
    const ex = Math.abs(dx) + Math.abs(Math.sin(th)) * t / 2, ey = Math.abs(dy) + Math.abs(Math.cos(th)) * t / 2
    return { id, diagonal: true, ax: a[0], ay: a[1], bx: b[0], by: b[1], x0: cx - ex, y0: cy - ey, x1: cx + ex, y1: cy + ey,
             horizontal: Math.abs(dx) >= Math.abs(dy), length: len, thickness: t, added: true }
  }

  // a railing is always straight along an axis (that is how the 3D draws it)
  const makeRail = (a: [number, number], b: [number, number]) => {
    const h = Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1])
    const lo = h ? Math.min(a[0], b[0]) : Math.min(a[1], b[1]), hi = h ? Math.max(a[0], b[0]) : Math.max(a[1], b[1])
    if (hi - lo < 0.3) return null
    const c = h ? a[1] : a[0], t = 0.05
    return h
      ? { id: baseId.rail + newRailings.length, x0: lo, y0: c - t, x1: hi, y1: c + t, horizontal: true, length: hi - lo, thickness: 2 * t, added: true }
      : { id: baseId.rail + newRailings.length, x0: c - t, y0: lo, x1: c + t, y1: hi, horizontal: false, length: hi - lo, thickness: 2 * t, added: true }
  }
  // a deck is a rectangle; "side" says which edge lies against the building (found from the nearest wall)
  const makeDeck = (a: [number, number], b: [number, number]) => {
    const x0 = Math.min(a[0], b[0]), x1 = Math.max(a[0], b[0]), y0 = Math.min(a[1], b[1]), y1 = Math.max(a[1], b[1])
    if (x1 - x0 < 0.6 || y1 - y0 < 0.6) return null
    const edges: Record<string, [number, number][]> = {   // sample points along each edge
      top: [[x0, y0], [(x0 + x1) / 2, y0], [x1, y0]], bottom: [[x0, y1], [(x0 + x1) / 2, y1], [x1, y1]],
      left: [[x0, y0], [x0, (y0 + y1) / 2], [x0, y1]], right: [[x1, y0], [x1, (y0 + y1) / 2], [x1, y1]],
    }
    let best = 'top', bd = Infinity
    for (const [name, pts] of Object.entries(edges)) {
      let d = 0
      for (const q of pts) {
        let m1 = Infinity
        for (const w of m.walls as W[]) m1 = Math.min(m1, Math.hypot(Math.max(w.x0 - q[0], 0, q[0] - w.x1), Math.max(w.y0 - q[1], 0, q[1] - w.y1)))
        d += m1
      }
      if (d < bd) { bd = d; best = name }
    }
    const side = ({ top: 'bottom', bottom: 'top', left: 'right', right: 'left' } as Record<string, string>)[best]
    return { id: baseId.deck + newDecks.length, x0, y0, x1, y1, side, roofed: deckRoof, added: true }
  }

  const down = (e: React.PointerEvent) => {
    const p = toPlan(e)
    if (tool === 'wall' || tool === 'railing' || tool === 'deck') {
      const s = snapPoint(p)
      setDrag({ a: s, b: s })
      ;(e.target as Element).setPointerCapture?.(e.pointerId)
    } else if (tool === 'erase') {
      const hitO = [...newOpenings].reverse().find(o => {
        const c = o.horizontal ? [(o.start + o.end) / 2, o.centre] : [o.centre, (o.start + o.end) / 2]
        return dist(p[0], p[1], c[0], c[1]) < Math.max(o.width / 2, 0.3)
      })
      if (hitO) { snapshot(); setNewOpenings(newOpenings.filter(o => o !== hitO)); return }
      const hitD = [...newDecks].reverse().find(d => p[0] >= d.x0 && p[0] <= d.x1 && p[1] >= d.y0 && p[1] <= d.y1)
      const hitR = [...newRailings].reverse().find(r => p[0] >= r.x0 - .2 && p[0] <= r.x1 + .2 && p[1] >= r.y0 - .2 && p[1] <= r.y1 + .2)
      if (hitR) { snapshot(); setNewRailings(newRailings.filter(r => r !== hitR)); return }
      const hitW = [...newWalls].reverse().find(w => p[0] >= w.x0 - .05 && p[0] <= w.x1 + .05 && p[1] >= w.y0 - .05 && p[1] <= w.y1 + .05)
      if (!hitW && hitD) { snapshot(); setNewDecks(newDecks.filter(d => d !== hitD)); return }
      if (hitW) {
        snapshot()
        setNewWalls(newWalls.filter(w => w !== hitW))
        setNewOpenings(newOpenings.filter(o => !(o.wallRef === hitW.id)))
      }
    } else {
      const o = openingFor(tool, p)
      if (o) {
        snapshot()
        o.id = baseId.op + newOpenings.length
        const hit = wallAt(p)
        if (hit && hit.w.added) o.wallRef = hit.w.id
        setNewOpenings([...newOpenings, o])
      }
    }
  }
  const move = (e: React.PointerEvent) => {
    const p = toPlan(e)
    if (drag) { const b = snapPoint(p, drag.a, tool === 'deck' ? true : tool === 'railing' ? false : e.altKey); setDrag({ ...drag, b }); setHover(b) } else setHover(p)
  }
  const up = () => {
    if (!drag) return
    const a = drag.a, b = drag.b
    setDrag(null)
    if (tool === 'railing') { const r = makeRail(a, b); if (r) { snapshot(); setNewRailings([...newRailings, r]) } return }
    if (tool === 'deck') { const d = makeDeck(a, b); if (d) { snapshot(); setNewDecks([...newDecks, d]) } return }
    const w = makeWall(a, b, wallFt * FT)
    if (w) { snapshot(); setNewWalls([...newWalls, w]) }
  }

  const px = (v: number) => v * ppm
  const ghost = tool === 'door' || tool === 'window' ? (hover ? openingFor(tool, hover) : null) : null
  const total = newWalls.length + newOpenings.length + newRailings.length + newDecks.length

  const drawOpening = (o: O, key: string | number, ghostMode = false) => {
    const col = o.type === 'door' ? '#b5651d' : '#2b9fd8'
    const [x0, y0, x1, y1] = o.horizontal
      ? [o.start, o.centre - o.thickness / 2, o.end, o.centre + o.thickness / 2]
      : [o.centre - o.thickness / 2, o.start, o.centre + o.thickness / 2, o.end]
    return <rect key={key} x={px(x0)} y={px(y0)} width={px(x1 - x0)} height={px(y1 - y0)} fill={col}
                 fillOpacity={ghostMode ? 0.45 : 0.9} stroke="#000" strokeOpacity={0.5} strokeWidth={1} />
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 50, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ background: '#fff', color: '#111', borderRadius: 10, width: 'min(1200px, 96vw)', height: '92vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', padding: '8px 12px', borderBottom: '1px solid #ddd' }}>
          <strong>Add walls, doors &amp; windows</strong>
          {(['wall', 'door', 'window', 'railing', 'deck', 'erase'] as Tool[]).map(t => (
            <button key={t} onClick={() => setTool(t)} style={{ fontWeight: tool === t ? 700 : 400, outline: tool === t ? '2px solid #2b6cb0' : 'none' }}>
              {({ wall: 'Wall', door: 'Door', window: 'Window', railing: 'Railing', deck: 'Deck', erase: 'Erase' } as Record<Tool, string>)[t]}
            </button>
          ))}
          <label>Wall&nbsp;thickness (ft) <input type="number" step="0.05" min="0.2" style={{ width: 64 }} value={wallFt} onChange={e => setWallFt(Number(e.target.value) || 0.5)} /></label>
          <label>Door (ft) <input type="number" step="0.1" min="2" style={{ width: 56 }} value={doorFt} onChange={e => setDoorFt(Number(e.target.value) || 3.5)} /></label>
          <label>Window (ft) <input type="number" step="0.1" min="1" style={{ width: 56 }} value={winFt} onChange={e => setWinFt(Number(e.target.value) || 4)} /></label>
          <label><input type="checkbox" checked={deckRoof} onChange={e => setDeckRoof(e.target.checked)} /> Roofed deck</label>
          <label>Zoom <input type="range" min="0.5" max="3" step="0.1" value={zoom} onChange={e => setZoom(Number(e.target.value))} /></label>
          <button onClick={undo} disabled={!hist.length}>Undo</button>
          <span style={{ flex: 1 }} />
          <button onClick={onClose}>Cancel</button>
          <button disabled={!total} onClick={() => onApply(newWalls, newOpenings, { railings: newRailings, decks: newDecks })} style={{ fontWeight: 700 }}>
            Add to 3D model{total ? ` (${[newWalls.length && `${newWalls.length} walls`, newOpenings.length && `${newOpenings.length} openings`, newRailings.length && `${newRailings.length} railings`, newDecks.length && `${newDecks.length} decks`].filter(Boolean).join(', ')})` : ''}
          </button>
        </div>
        <div style={{ padding: '4px 12px', fontSize: 12, color: '#555' }}>
          {tool === 'wall' && 'Drag to draw a wall. It snaps to wall ends and stays straight; hold Alt for an angled wall.'}
          {tool === 'door' && 'Click on a wall to place a door.'}
          {tool === 'window' && 'Click on a wall to place a window.'}
          {tool === 'railing' && 'Drag to draw a railing (straight, horizontal or vertical). Decks already get their own balustrade.'}
          {tool === 'deck' && 'Drag a rectangle for the deck. Draw it against the building; the side touching the walls becomes the building side.'}
          {tool === 'erase' && 'Click something you added to remove it (detected walls cannot be erased here).'}
        </div>
        <div style={{ flex: 1, overflow: 'auto', background: '#eee' }}>
          <div style={{ position: 'relative', width: `${zoom * 100}%`, margin: '0 auto' }}>
            <img ref={imgRef} src={image} alt="plan" draggable={false} style={{ display: 'block', width: '100%' }}
                 onLoad={e => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
            <svg ref={svgRef} viewBox={`0 0 ${size.w} ${size.h}`}
                 style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none', cursor: tool === 'erase' ? 'pointer' : 'crosshair' }}
                 onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerLeave={() => setHover(null)}>
              {newWalls.map(w => w.diagonal
                ? <line key={w.id} x1={px(w.ax)} y1={px(w.ay)} x2={px(w.bx)} y2={px(w.by)} stroke="#d62828" strokeWidth={px(w.thickness)} />
                : <rect key={w.id} x={px(w.x0)} y={px(w.y0)} width={px(w.x1 - w.x0)} height={px(w.y1 - w.y0)} fill="#d62828" fillOpacity={0.85} />)}
              {newDecks.map(d => <g key={`d${d.id}`}>
                <rect x={px(d.x0)} y={px(d.y0)} width={px(d.x1 - d.x0)} height={px(d.y1 - d.y0)} fill="#8b6a47" fillOpacity={0.55} stroke="#5c3b1e" strokeWidth={2} strokeDasharray="8 4" />
                <text x={px((d.x0 + d.x1) / 2)} y={px((d.y0 + d.y1) / 2)} textAnchor="middle" fontSize={Math.max(12, size.w / 80)} fill="#fff">
                  Deck {((d.x1 - d.x0) / FT).toFixed(0)}×{((d.y1 - d.y0) / FT).toFixed(0)} ft</text></g>)}
              {newRailings.map(r => <rect key={`r${r.id}`} x={px(r.x0)} y={px(r.y0)} width={px(r.x1 - r.x0)} height={px(r.y1 - r.y0)} fill="#303740" />)}
              {newOpenings.map((o, i) => drawOpening(o, i))}
              {ghost && drawOpening(ghost, 'g', true)}
              {drag && tool === 'deck' && (() => { const d = makeDeck(drag.a, drag.b)
                return d ? <rect x={px(d.x0)} y={px(d.y0)} width={px(d.x1 - d.x0)} height={px(d.y1 - d.y0)} fill="#8b6a47" fillOpacity={0.4} stroke="#5c3b1e" strokeWidth={2} /> : null })()}
              {drag && tool === 'railing' && (() => { const r = makeRail(drag.a, drag.b)
                return r ? <rect x={px(r.x0)} y={px(r.y0)} width={px(r.x1 - r.x0)} height={px(r.y1 - r.y0)} fill="#303740" fillOpacity={0.6} /> : null })()}
              {drag && tool === 'wall' && (() => { const w = makeWall(drag.a, drag.b, wallFt * FT)
                return w ? (w.diagonal
                  ? <line x1={px(w.ax)} y1={px(w.ay)} x2={px(w.bx)} y2={px(w.by)} stroke="#d62828" strokeOpacity={0.5} strokeWidth={px(w.thickness)} />
                  : <rect x={px(w.x0)} y={px(w.y0)} width={px(w.x1 - w.x0)} height={px(w.y1 - w.y0)} fill="#d62828" fillOpacity={0.5} />) : null })()}
              {drag && <text x={px(drag.b[0]) + 8} y={px(drag.b[1]) - 8} fontSize={Math.max(12, size.w / 80)} fill="#d62828">
                {(dist(drag.a[0], drag.a[1], drag.b[0], drag.b[1]) / FT).toFixed(1)} ft</text>}
              {(tool === 'wall' || tool === 'railing' || tool === 'deck') && hover && !drag && <circle cx={px(snapPoint(hover)[0])} cy={px(snapPoint(hover)[1])} r={Math.max(3, size.w / 300)} fill="none" stroke="#d62828" />}
            </svg>
          </div>
        </div>
      </div>
    </div>
  )
}