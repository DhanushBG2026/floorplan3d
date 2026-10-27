import { useState, useEffect, useMemo} from 'react'
import { FT, toFt, toM, fmtFtIn } from './units'
import Viewer from './Viewer'
import type { PlanModel } from './types'
import './App.css'
import { GROUNDS, type GroundType } from './ground'
import type { DimMode } from './types'
import { buildGroups } from './groups'
import {
  convertPlan,
  previewPlan,
  DEFAULT_ADJUSTMENTS,
  type ImageAdjustments,
  type PreviewResult,
  type PageThumb,

} from './api'
import CropDialog, { type CropBox } from './CropDialog'
import ChatPanel from './ChatPanel'
import type { RoofType } from './roof'
import { WALL_FINISHES, ROOF_FINISHES, DEFAULT_ROOF_COLOR, type WallFinishKey, type RoofFinishKey } from './materials'
import { applyExtensions, groupAxis, snapCoordinate, clampExt, type Ext } from './extend'
import PlanEditor from './PlanEditor'
import SiteMap from './SiteMap'
import type { SiteConfig } from './site'
import FloorMarker from './FloorMarker'
import type { ExtraSel } from './Viewer'
import PagePicker from './PagePicker' 

type DimensionUnit = 'ft-in' | 'm' | 'cm' | 'mm' | 'in'

interface DeleteHistory { past: number[][]; present: number[]; future: number[][] }
const EMPTY_HIST: DeleteHistory = { past: [], present: [], future: [] }

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [scale, setScale] = useState(100)
  const [knownW, setKnownW] = useState('')
  const [ppm, setPpm] = useState('')
  const [wallH, setWallH] = useState(9)
  const [dimMode, setDimMode] = useState<DimMode>('overall')
  const [dimensionUnit, setDimensionUnit] = useState<DimensionUnit>('ft-in')
  const [model, setModel] = useState<PlanModel | null>(null)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [ground, setGround] = useState<GroundType>('grass')
  const [heights, setHeights] = useState<Record<number, number>>({})  
  const [thicknesses, setThicknesses] = useState<Record<number, number>>({})
  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [hist, setHist] = useState<DeleteHistory>(EMPTY_HIST)
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [pages, setPages] = useState<PageThumb[]>([])
  const [pageNo, setPageNo] = useState(-1)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [crop, setCrop] = useState<CropBox | null>(null)
  const [cropOpen, setCropOpen] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const deleted = hist.present
  const groups = useMemo(() => (model ? buildGroups(model) : null), [model])
  const [docText, setDocText] = useState('')
  const [adjustments, setAdjustments] = useState<ImageAdjustments>({ ...DEFAULT_ADJUSTMENTS })
  const [roofType, setRoofType] = useState<RoofType | 'design'>('none')
  const [roofOn, setRoofOn] = useState(true)
  const [ridgeFt, setRidgeFt] = useState(10.8)
  const [overhangFt, setOverhangFt] = useState(1)
  const [stumpFt, setStumpFt] = useState(0)
  const [sillFt, setSillFt] = useState(2.95)
  const [topFt, setTopFt] = useState(6.9)
  const [wallFinish, setWallFinish] = useState<WallFinishKey>('plaster')
  const [roofFinish, setRoofFinish] = useState<RoofFinishKey>('metal')
  const [roofColor, setRoofColor] = useState(DEFAULT_ROOF_COLOR)
  const [detector, setDetector] = useState<'opencv' | 'ml'>('opencv')
  const [extensions, setExtensions] = useState<Record<number, Ext>>({})
  const [snapOn, setSnapOn] = useState(true)
  const [showPlan, setShowPlan] = useState(true)
  const [extColor, setExtColor] = useState('')   // '' = use the finish's own colour
  const [intColor, setIntColor] = useState('')
  const [stackOn, setStackOn] = useState(false)
  const [floorOrder, setFloorOrder] = useState<number[]>([])
  const [floorNudge, setFloorNudge] = useState<Record<number, [number, number]>>({})
  const [editorOpen, setEditorOpen] = useState(false)
  const [site, setSite] = useState<SiteConfig | null>(null)
  const [siteOpen, setSiteOpen] = useState(false)
  const [floorsOpen, setFloorsOpen] = useState(false)
  const [extra, setExtra] = useState<ExtraSel | null>(null)
  const [removed, setRemoved] = useState<{ walls: any[]; openings: any[]; railings: any[]; decks: any[] }[]>([])

  const pickExtra = (s: ExtraSel | null) => {setExtra(s); if (s) handleSelect(null) }

  const deleteExtra = () => {
    if (!extra || !model) return
    const { kind, id } = extra
    const m: any = model
    const gone = { walls: [] as any[], openings: [] as any[], railings: [] as any[], decks: [] as any[] }
    const keep = (arr: any[], bucket: any[], drop: (x: any) => boolean) => arr.filter(x => (drop(x) ? (bucket.push(x), false) : true))
    const next: any = { ...m }
    if (kind === 'wall') {
      next.walls = keep(m.walls, gone.walls, w => w.id === id)
      next.openings = keep(m.openings, gone.openings, o => o.wallRef === id)     // doors/windows drawn on that wall
    }
    if (kind === 'opening') next.openings = keep(m.openings, gone.openings, o => o.id === id)
    if (kind === 'railing') next.railings = keep(m.railings ?? [], gone.railings, r => r.id === id)
    if (kind === 'deck') next.decks = keep(m.decks ?? [], gone.decks, d => d.id === id)
    setRemoved(h => [...h, gone])
    setModel(next)
    setExtra(null)
  }
  const restoreExtra = () => {
    const last = removed[removed.length - 1]
    if (!last) return
    setModel(m => m ? ({ ...m, walls: [...m.walls, ...last.walls], openings: [...m.openings, ...last.openings],
      railings: [...(m.railings ?? []), ...last.railings], decks: [...((m as any).decks ?? []), ...last.decks] } as any) : m)
    setRemoved(h => h.slice(0, -1))
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!extra) return
      if (/INPUT|TEXTAREA|SELECT/.test((e.target as HTMLElement)?.tagName ?? '')) return
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); deleteExtra() }
      if (e.key === 'Escape') setExtra(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })


    const applySuggestion = (p: PreviewResult) => {
    const sg = p.suggest
    if (!sg || sg.page !== p.page) return
    if (sg.crop) setCrop(sg.crop as CropBox)
    if (sg.scale && sg.scale > 0) setScale(Math.round(sg.scale * 10) / 10)
  }

  const onFileChosen = async (f: File | null) => {
    setFile(f); setPreview(null); setCrop(null); setDocText(''); setPages([]); setPageNo(-1)
    if (!f) return
    setPreviewing(true)
    setStatus('Loading preview…')
    try {
      const p = await previewPlan(f)
      setPreview(p)
      setDocText(p.text ?? '')
      setPageNo(p.page ?? -1)
      if ((p.pageList?.length ?? 0) > 1) {
        setPages(p.pageList!)
        setPickerOpen(true)
        setStatus('Choose the page with the floor plan.')
      } else {
        applySuggestion(p)
        setCropOpen(true)
        setStatus('Select the floor plan area.')
      }
    } catch (e) {
      setStatus('Error: ' + (e as Error).message)
    }
    setPreviewing(false)
  }

  const choosePage = async (n: number) => {
    setPickerOpen(false)
    if (!file) return
    if (preview && n === preview.page) {
      applySuggestion(preview)
      setCropOpen(true)
      setStatus('Select the floor plan area.')
      return
    }
    setPreviewing(true)
    setStatus('Loading page…')
    try {
      const p = await previewPlan(file, n)
      setPreview(p)
      setPageNo(n)
      setCrop(null)
      applySuggestion(p)
      setCropOpen(true)
      setStatus('Select the floor plan area.')
    } catch (e) {
      setStatus('Error: ' + (e as Error).message)
    }
    setPreviewing(false)
  }
  

  const generate = async () => {
    if (!file) { setStatus('Choose a PDF or image first.'); return }
    setLoading(true)
    setStatus('Analysing…')
    try {
      const m = await convertPlan(file, {
        drawingScale: scale,
        detector,
        knownWidthM: toM(parseFloat(knownW) || 0),
        pxPerM: (parseFloat(ppm) || 0) / FT,
        crop,
        adjustments,
        page: pageNo,
      })
      setModel(m)
      setFloorOrder(m.floors?.map(f => f.id) ?? [])
      setFloorNudge({})
      setStackOn(!!m.floors && m.floors.length > 1 && m.floors.every(f => f.similar !== false))
      const d = m.design
      if (d?.roof) {
        setRoofType('design')
        setRoofOn(true)
        if (d.levels?.wallTop) setWallH(Number(toFt(d.levels.wallTop).toFixed(2)))
        setStumpFt(prev => prev || 2)          // deck needs a raised floor; tweak as you like
      } else if (roofType === 'design') {
        setRoofType('none')
      }
      setHeights({})
      setThicknesses({})
      setHeights({})
      setExtensions({})
      setThicknesses({})
      setHist(EMPTY_HIST)
      setSelectedIds([])
      const n = (t: string) => m.openings.filter(o => o.type === t).length
      setStatus(`${m.walls.length} walls, ${n('door')} doors, ${n('window')} windows`)
    } catch (e) {
      setStatus('Error: ' + (e as Error).message)
    }
    setLoading(false)
  }

  const groupOfWall = (id: number) => groups?.groupOf[id] ?? id
  const membersOf = (gid: number) => groups?.members[gid] ?? [gid]

  const selectedGroupIds = Array.from(new Set(selectedIds.map(groupOfWall)))
  const selectedWall =
    selectedGroupIds.length === 1
      ? model?.walls.find(w => w.id === selectedGroupIds[0]) ?? null
      : null

  // selecting any piece selects the whole wall; ctrl+click toggles the whole wall
  const handleSelect = (id: number | null, additive = false) => {
    if (id === null) { setSelectedIds([]); return }
    const ids = membersOf(groupOfWall(id))
    setSelectedIds(prev => {
      if (!additive) return ids
      const allIn = ids.every(i => prev.includes(i))
      return allIn ? prev.filter(i => !ids.includes(i)) : Array.from(new Set([...prev, ...ids]))
    })
  }


    const selectedHeight = selectedWall ? (heights[selectedWall.id] ?? toM(wallH)) : 0
    const selectedThickness = selectedWall ? (thicknesses[selectedWall.id] ?? selectedWall.thickness) : 0

    const clampH = (v: number) => Math.min(10, Math.max(0.3, v))

    // set the same height on every selected wall
    const setSelectedHeight = (v: number) => {
      if (!Number.isFinite(v)) return
      const val = clampH(v)
      setHeights(h => {
        const n = { ...h }
        selectedIds.forEach(id => { n[id] = val })
        return n
      })
    }

    // raise / lower every selected wall relative to its own current height
    const nudgeSelectedHeight = (delta: number) =>
      setHeights(h => {
        const n = { ...h }
        selectedIds.forEach(id => { n[id] = clampH((h[id] ?? toM(wallH)) + delta) })
        return n
      })

    const resetSelectedHeights = () =>
      setHeights(h => {
        const n = { ...h }
        selectedIds.forEach(id => { delete n[id] })
        return n
      })

    const multiHeights = selectedGroupIds.map(g => heights[g] ?? toM(wallH))
    const heightsMixed = multiHeights.some(v => Math.abs(v - multiHeights[0]) > 0.005)
    const multiHeight = multiHeights[0] ?? 0

    const setWallHeight = (gid: number, v: number) => {
    if (!Number.isFinite(v)) return
    const val = Math.min(10, Math.max(0.3, v))
    setHeights(h => {
      const n = { ...h }
      membersOf(gid).forEach(id => { n[id] = val })
      return n
    })
  }

  const setWallThickness = (gid: number, v: number) => {
    if (!Number.isFinite(v)) return
    const val = Math.min(1, Math.max(0.05, v))
    setThicknesses(t => {
      const n = { ...t }
      membersOf(gid).forEach(id => { n[id] = val })
      return n
    })
  }

  const resetWall = (gid: number) => {
    setHeights(h => {
      const n = { ...h }
      membersOf(gid).forEach(id => { delete n[id] })
      return n
    })
    setExtensions(x => { const n = { ...x }; delete n[gid]; return n })
    setThicknesses(t => {
      const n = { ...t }
      membersOf(gid).forEach(id => { delete n[id] })
      return n
    })
  }

    const axis = model && groups && selectedWall ? groupAxis(model.walls, groups, selectedWall.id) : null
  const selExt: Ext = selectedWall ? (extensions[selectedWall.id] ?? { start: 0, end: 0 }) : { start: 0, end: 0 }

  const setExtension = (gid: number, e: Ext) => setExtensions(x => ({ ...x, [gid]: e }))

  const setEnd = (which: 'start' | 'end', metres: number) => {
    if (!selectedWall || !axis || !Number.isFinite(metres)) return
    setExtension(selectedWall.id, clampExt(axis, { ...selExt, [which]: metres }))
  }

  // pull both ends to the nearest wall within 1.5 m
  const snapEnds = () => {
    if (!selectedWall || !axis || !model || !groups || axis.diagonal) return
    const ws = applyExtensions(model.walls, groups, extensions).filter(w => !deleted.includes(w.id))
    const skip = groups.members[selectedWall.id] ?? []
    const a = snapCoordinate({ walls: ws, skip, horizontal: axis.horizontal, perp: axis.perp, value: axis.loBase - selExt.start, threshold: 1.5 })
    const z = snapCoordinate({ walls: ws, skip, horizontal: axis.horizontal, perp: axis.perp, value: axis.hiBase + selExt.end, threshold: 1.5 })
    setExtension(selectedWall.id, clampExt(axis, { start: axis.loBase - a.value, end: z.value - axis.hiBase }))
  }

      const deleteWalls = (ids: number[]) => {
    if (ids.length === 0) return
    setHist(h => {
      const next = Array.from(new Set([...h.present, ...ids]))
      if (next.length === h.present.length) return h
      return { past: [...h.past, h.present], present: next, future: [] }
    })
    setSelectedIds([])
  }

  const restoreAll = () =>
    setHist(h =>
      h.present.length ? { past: [...h.past, h.present], present: [], future: [] } : h,
    )

  const undo = () => {
    setHist(h =>
      h.past.length
        ? { past: h.past.slice(0, -1), present: h.past[h.past.length - 1], future: [h.present, ...h.future] }
        : h,
    )
    setSelectedIds([])
  }

  const redo = () => {
    setHist(h =>
      h.future.length
        ? { past: [...h.past, h.present], present: h.future[0], future: h.future.slice(1) }
        : h,
    )
    setSelectedIds([])
  }

  const canUndo = hist.past.length > 0
  const canRedo = hist.future.length > 0

  // Delete/Backspace deletes the selection; Ctrl+Z undo; Ctrl+Y or Ctrl+Shift+Z redo
  useEffect(() => {
      const onKey = (e: KeyboardEvent) => {
        const tag = (e.target as HTMLElement).tagName
        if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
        const mod = e.ctrlKey || e.metaKey
        const k = e.key.toLowerCase()
        if (mod && k === 'z') {
          e.preventDefault()
          if (e.shiftKey) redo()
          else undo()
        } else if (mod && k === 'y') {
          e.preventDefault()
          redo()
        } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedIds.length > 0) {
          e.preventDefault()
          deleteWalls(selectedIds)
        }
      }
      window.addEventListener('keydown', onKey)
      return () => window.removeEventListener('keydown', onKey)
    }, [selectedIds])

   // the order always matches the floors that exist: stale ids are dropped, new ones added
  const orderedIds = (() => {
    const ids = (model?.floors ?? []).map(f => f.id)
    const kept = floorOrder.filter(id => ids.includes(id))
    return [...kept, ...ids.filter(id => !kept.includes(id))]
  })()

  const stack = model?.floors && model.floors.length > 1 && stackOn
    ? { order: floorOrder, nudge: floorNudge } : null
  const LEVELS = ['Ground floor', 'First floor', 'Second floor', 'Third floor', 'Fourth floor']
  const moveFloor = (i: number, d: number) =>
    setFloorOrder(o => { const n = [...o]; [n[i], n[i + d]] = [n[i + d], n[i]]; return n })
  const nudge = (id: number, axis: 0 | 1, metres: number) =>
    setFloorNudge(n => { const c: [number, number] = [...(n[id] ?? [0, 0])] as [number, number]; c[axis] = metres; return { ...n, [id]: c } })
  const planName = (id: number) => {
    const fl = model?.floors ?? [], f = fl.find(x => x.id === id)
    if (!f) return `Floor ${id + 1}`
    const cx = fl.map(g => g.bbox.x0 + g.bbox.x1), cy = fl.map(g => g.bbox.y0 + g.bbox.y1)
    const horiz = Math.max(...cx) - Math.min(...cx) >= Math.max(...cy) - Math.min(...cy)
    const vals = horiz ? cx : cy
    const mine = horiz ? f.bbox.x0 + f.bbox.x1 : f.bbox.y0 + f.bbox.y1
    const rank = [...vals].sort((a, b) => a - b).indexOf(mine)
    const names = horiz ? ['left', 'middle', 'right'] : ['top', 'middle', 'bottom']
    return `Plan on the ${rank === 0 ? names[0] : rank === fl.length - 1 ? names[2] : names[1]}`
  }

  const applyEdits = (ws: any[], os: any[], extra: { railings: any[]; decks: any[] }) => {
    setModel(m => {
      if (!m) return m
      const next: any = {
        ...m,
        walls: [...m.walls, ...ws],
        openings: [...m.openings, ...os],
        railings: [...(m.railings ?? []), ...extra.railings],
        decks: [...((m as any).decks ?? []), ...extra.decks],
      }
      if (next.floors) {
        const at = (x: number, y: number) => next.floors.find((f: any) => x >= f.bbox.x0 && x <= f.bbox.x1 && y >= f.bbox.y0 && y <= f.bbox.y1)
        next.floors = next.floors.map((f: any) => ({ ...f, wallIds: [...f.wallIds], openingIds: [...f.openingIds], railingIds: [...f.railingIds] }))
        for (const w of ws) at((w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2)?.wallIds.push(w.id)
        for (const r of extra.railings) at((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)?.railingIds.push(r.id)
        for (const o of os) {
          const [x, y] = o.horizontal ? [(o.start + o.end) / 2, o.centre] : [o.centre, (o.start + o.end) / 2]
          at(x, y)?.openingIds.push(o.id)
        }
      }
      return next
    })
    setEditorOpen(false)
  }

  const detectFloors = async (boxes: number[][], labels: string[], count: number) => {
    const r = await fetch('/api/floors', {          // use the same base URL as your other api.ts calls
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: { walls: model!.walls, openings: model!.openings, stairs: model!.stairs, railings: model!.railings },
        boxes, labels, count,
      }),
    })
    if (!r.ok) throw new Error(await r.text())
    return (await r.json()).floors
  }
  

  return (
    <div className="app">
      <aside className="side">
        <h1>Floor plan → 3D</h1>

        <label>Floor plan (PDF, PNG, JPG)</label>
        <input type="file" accept=".pdf,image/*" onChange={e => onFileChosen(e.target.files?.[0] ?? null)} />
        {preview && (
          <div className="crop-summary">
            <span>{crop ? 'Plan area selected' : 'Using full page'}</span>
            <button className="link" onClick={() => setCropOpen(true)}>Change area</button>
            {pages.length > 1 && (
              <button className="link" onClick={() => setPickerOpen(true)}>Change page</button>
            )}
          </div>
        )}

        <label>PDF drawing scale (1 : N)</label>
        <input type="number" min={1} value={scale} onChange={e => setScale(Number(e.target.value))} />

        <label>Real overall plan width in feet (optional)</label>
        <input type="number" step="0.1" placeholder="e.g. 40" value={knownW}
               onChange={e => setKnownW(e.target.value)} />

        <label>Pixels per foot (images only, optional)</label>
        <input type="number" step="0.1" value={ppm} onChange={e => setPpm(e.target.value)} />

        <label>Wall height (ft)</label>
        <input type="number" step="0.5" value={wallH} onChange={e => setWallH(Number(e.target.value) || 9)} />

        <h3 className="sub">Roof and levels</h3>

        <label>Roof</label>
        <select value={roofType} onChange={e => setRoofType(e.target.value as RoofType | 'design')}>
          {model?.design?.roof && <option value="design">From PDF ({model.design.roof.type})</option>}
          <option value="none">None</option>
          <option value="gable">Gable</option>
          <option value="hip">Hip</option>
          <option value="skillion">Skillion</option>
          <option value="flat">Flat</option>
        </select>

        <label>Wall finish</label>
        <select value={wallFinish} onChange={e => setWallFinish(e.target.value as WallFinishKey)}>
          {Object.entries(WALL_FINISHES).map(([key, f]) => (
            <option key={key} value={key}>{f.label}</option>
          ))}
        </select>

        <label>Exterior wall colour</label>
        <div className="row">
          <input type="color" value={extColor || '#ffffff'} onChange={e => setExtColor(e.target.value)} />
          <button className="secondary" onClick={() => setExtColor('')} disabled={!extColor}>Reset</button>
        </div>
        <label>Interior wall colour</label>
        <div className="row">
          <input type="color" value={intColor || '#ffffff'} onChange={e => setIntColor(e.target.value)} />
          <button className="secondary" onClick={() => setIntColor('')} disabled={!intColor}>Reset</button>
        </div>

        {roofType !== 'none' && (
          <>
            <label className="check">
              <input type="checkbox" checked={roofOn} onChange={e => setRoofOn(e.target.checked)} />
              Show roof
            </label>
            <label>Roof finish</label>
            <select value={roofFinish} onChange={e => setRoofFinish(e.target.value as RoofFinishKey)}>
              {Object.entries(ROOF_FINISHES).map(([key, f]) => (
                <option key={key} value={key}>{f.label}</option>
              ))}
            </select>
            <label>Roof colour</label>
            <input type="color" value={roofColor} onChange={e => setRoofColor(e.target.value)} />
            {roofType !== 'design' && (
              <>
                <label>Ridge or high point, above floor (ft)</label>
                <input type="number" step="0.1" value={ridgeFt} onChange={e => setRidgeFt(Number(e.target.value) || 0)} />
              </>
            )}
            <label>Roof overhang (ft)</label>
            <input type="number" step="0.1" min={0} value={overhangFt} onChange={e => setOverhangFt(Number(e.target.value) || 0)} />
          </>
        )}

        <label>Floor above ground, stumps (ft)</label>
        <input type="number" step="0.1" min={0} value={stumpFt} onChange={e => setStumpFt(Number(e.target.value) || 0)} />

        <label>Window sill height (ft)</label>
        <input type="number" step="0.1" min={0} value={sillFt} onChange={e => setSillFt(Number(e.target.value) || 0)} />

        <label>Window and door top (ft)</label>
        <input type="number" step="0.1" min={1} value={topFt} onChange={e => setTopFt(Number(e.target.value) || 1)} />

        <label>Ground</label>
        <select value={ground} onChange={e => setGround(e.target.value as GroundType)}>
          {Object.entries(GROUNDS).map(([key, g]) => (
            <option key={key} value={key}>{g.label}</option>
          ))}
        </select>

        <label>
          Detection method
          <select value={detector} onChange={e => setDetector(e.target.value as 'opencv' | 'ml')}>
            <option value="ml">ML model</option>
            <option value="opencv">OpenCV</option>
          </select>
        </label>

        <button onClick={generate} disabled={loading || previewing}>Generate 3D model</button>
        <div className="status">{status}</div>

        <button onClick={() => setEditorOpen(true)} disabled={!model}>Add walls / doors / windows</button>

        {model?.plan && (
          <label className="check">
            <input type="checkbox" checked={showPlan} onChange={e => setShowPlan(e.target.checked)} />
            Show plan image on floor
          </label>
        )}

        {model?.floors && model.floors.length > 1 && (
          <div className="panel">
            <h2>{model.floors.length} floor plans found</h2>
            <label className="check">
              <input type="checkbox" checked={stackOn} onChange={e => setStackOn(e.target.checked)} />
              Stack them as floors
            </label>
            {stackOn && (
              <>
                <div className="meta">Bottom to top</div>
                {orderedIds.map((id, i) => (
                  <div key={id}>
                    <div className="row">
                      <span style={{ flex: 1 }}>{LEVELS[i] ?? `Floor ${i + 1}`}: {planName(id)}</span>
                      <button className="secondary" disabled={i === 0} onClick={() => moveFloor(i, -1)}>▲</button>
                      <button className="secondary" disabled={i === orderedIds.length - 1} onClick={() => moveFloor(i, 1)}>▼</button>
                    </div>
                    {i > 0 && (
                      <>
                        <label>Shift left/right (ft)</label>
                        <input type="range" min={-10} max={10} step={0.1}
                               value={toFt((floorNudge[id] ?? [0, 0])[0])}
                               onChange={e => nudge(id, 0, toM(parseFloat(e.target.value)))} />
                        <label>Shift up/down on plan (ft)</label>
                        <input type="range" min={-10} max={10} step={0.1}
                               value={toFt((floorNudge[id] ?? [0, 0])[1])}
                               onChange={e => nudge(id, 1, toM(parseFloat(e.target.value)))} />
                      </>
                    )}
                  </div>
                ))}
              </>
            )}
          </div>
        )}

        <label>Dimensions</label>
        <select value={dimMode} onChange={e => setDimMode(e.target.value as DimMode)}>
          <option value="off">Off</option>
          <option value="overall">Overall + doors/windows</option>
          <option value="all">All walls</option>
        </select>
        <label>Dimension units</label>
        <select
          value={dimensionUnit}
          onChange={e => setDimensionUnit(e.target.value as DimensionUnit)}
        >
          <option value="ft-in">Feet &amp; inches (ft/in)</option>
          <option value="m">Metres (m)</option>
          <option value="cm">Centimetres (cm)</option>
          <option value="mm">Millimetres (mm)</option>
          <option value="in">Inches (in)</option>
        </select>
        <div className="legend">
          <span className="dot window" /> Window
          <span className="dot door" /> Door
          <span className="dot opening" /> Opening
        </div>

        {selectedWall ? (
          <div className="panel">
            <h2>Wall #{selectedWall.id}</h2>
            <div className="meta">
                {fmtFtIn(groups?.length[selectedWall.id] ?? selectedWall.length)} long · {fmtFtIn(selectedThickness)} thick
            </div>
            <label>Height (ft)</label>
            <input
              type="number" step="0.1" min={1} max={30}
              value={Number(toFt(selectedHeight).toFixed(2))}
              onChange={e => setWallHeight(selectedWall.id, toM(parseFloat(e.target.value)))}
            />
            <input
              type="range" min={1} max={20} step={0.1}
              value={Math.min(toFt(selectedHeight), 20)}
              onChange={e => setWallHeight(selectedWall.id, toM(parseFloat(e.target.value)))}
            />
            <label>Thickness (ft)</label>
            <input
              type="number" step="0.05" min={0.2} max={3}
              value={Number(toFt(selectedThickness).toFixed(2))}
              onChange={e => setWallThickness(selectedWall.id, toM(parseFloat(e.target.value)))}
            />
            <input
              type="range" min={0.2} max={2} step={0.05}
              value={Math.min(toFt(selectedThickness), 2)}
              onChange={e => setWallThickness(selectedWall.id, toM(parseFloat(e.target.value)))}
            />

            {axis && (
              <>
                <label>Extend start (ft)</label>
                <input
                  type="number" step="0.1"
                  value={Number(toFt(selExt.start).toFixed(2))}
                  onChange={e => setEnd('start', toM(parseFloat(e.target.value)))}
                />
                <input
                  type="range" min={Math.floor(toFt(-(axis.startLen - 0.1)))} max={30} step={0.05}
                  value={toFt(selExt.start)}
                  onChange={e => setEnd('start', toM(parseFloat(e.target.value)))}
                />
                <label>Extend end (ft)</label>
                <input
                  type="number" step="0.1"
                  value={Number(toFt(selExt.end).toFixed(2))}
                  onChange={e => setEnd('end', toM(parseFloat(e.target.value)))}
                />
                <input
                  type="range" min={Math.floor(toFt(-(axis.endLen - 0.1)))} max={30} step={0.05}
                  value={toFt(selExt.end)}
                  onChange={e => setEnd('end', toM(parseFloat(e.target.value)))}
                />
                <label className="check">
                  <input type="checkbox" checked={snapOn} onChange={e => setSnapOn(e.target.checked)} />
                  Snap to nearby walls when dragging
                </label>
                {!axis.diagonal && (
                  <button className="secondary" onClick={snapEnds}>Snap ends to nearest walls</button>
                )}
              </>
            )}

            <button className="secondary" onClick={() => resetWall(selectedWall.id)}>
              Reset to default
            </button>
            <button className="danger" onClick={() => deleteWalls([selectedWall.id])}>
              Delete wall
            </button>
          </div>
        ) : selectedGroupIds.length > 1 ? (
          <div className="panel">
            <h2>{selectedGroupIds.length} walls selected</h2>
            <div className="meta">Ctrl+click to add or remove walls</div>

            <label>Height (ft){heightsMixed ? ' – mixed' : ''}</label>
            <input
              type="number" step="0.1" min={1} max={30}
              placeholder={heightsMixed ? 'Mixed' : undefined}
              value={heightsMixed ? '' : Number(toFt(multiHeight).toFixed(2))}
              onChange={e => setSelectedHeight(toM(parseFloat(e.target.value)))}
            />
            <input
              type="range" min={1} max={20} step={0.1}
              value={Math.min(toFt(multiHeight), 20)}
              onChange={e => setSelectedHeight(toM(parseFloat(e.target.value)))}
            />
            <div className="row">
              <button className="secondary" onClick={() => nudgeSelectedHeight(-toM(1))}>−1 ft</button>
              <button className="secondary" onClick={() => nudgeSelectedHeight(toM(1))}>+1 ft</button>
            </div>
            <button className="secondary" onClick={resetSelectedHeights}>Reset height</button>

            <button className="danger" onClick={() => deleteWalls(selectedIds)}>
              Delete {selectedGroupIds.length} walls
            </button>
            <button className="secondary" onClick={() => setSelectedIds([])}>
              Clear selection
            </button>

          </div>
        ) : (
        model && <div className="hint">Click a wall to select it. Ctrl+click to select several.</div>
      )}

      {deleted.length > 0 && (
        <button className="secondary" onClick={restoreAll}>
          Restore {deleted.length} deleted wall{deleted.length > 1 ? 's' : ''}
        </button>
      )}

      {extra && (
        <div>
          <h2>Selected {extra.kind}</h2>
          <div className="meta">Added in the editor</div>
          <button onClick={deleteExtra}>Delete {extra.kind}</button>
        </div>
      )}
      {removed.length > 0 && <button onClick={restoreExtra}>Restore last deleted addition</button>}
              
      {model && <img className="overlay" src={model.overlay} alt="detection overlay" />}
      </aside>

      <main className="view">
        {model && (
            <div className="toolbar">
              <button onClick={undo} disabled={!canUndo} title="Undo wall delete (Ctrl+Z)">↶ Undo</button>
              <button onClick={redo} disabled={!canRedo} title="Redo wall delete (Ctrl+Y)">↷ Redo</button>
              <button onClick={() => setSiteOpen(true)} disabled={!model}>Place on site (map)</button>
              <button onClick={() => setFloorsOpen(true)} disabled={!model}>Mark floor plans</button>
            </div>
          )}
          {model && groups ? (
          <Viewer
            model={model}
            wallH={toM(wallH)}
            heights={heights}
            thicknesses={thicknesses}
            deleted={deleted}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            groups={groups}
            dimMode={dimMode}
            dimensionUnit={dimensionUnit}
            ground={ground}
            roof={{ type: roofType as RoofType, visible: roofOn, ridgeM: toM(ridgeFt), overhangM: toM(overhangFt) }}
            stumpM={toM(stumpFt)}
            spec={{ sillM: toM(sillFt), headM: toM(topFt), doorM: toM(topFt) }}
            finish={{ wall: WALL_FINISHES[wallFinish], roof: ROOF_FINISHES[roofFinish], roofColor, exteriorColor: extColor || undefined, interiorColor: intColor || undefined }}
            extensions={extensions}
            onExtend={setExtension}
            snapOn={snapOn}
            stack={stack}
            showPlan={showPlan}
            site={site}
            selectedExtra={extra}
            onPickExtra={pickExtra}
          />
        ) : (
          <p className="empty">Upload a plan to begin</p>
        )}

        {site && (
          <div
            className="attrib"
            style={{
              position: 'absolute', right: 6, bottom: 4, zIndex: 5,
              fontSize: 10, color: '#fff', background: 'rgba(0,0,0,.45)',
              padding: '1px 6px', borderRadius: 3, pointerEvents: 'none',
            }}
          >
            {site.layer === 'satellite' ? 'Imagery © Esri, Maxar, Earthstar Geographics' : '© OpenStreetMap contributors'}
          </div>
        )}

        {preview && (
          <ChatPanel documentText={docText} docKey={file ? `${file.name}-${file.size}` : ''} />
        )}
        
      </main>
      {editorOpen && model && (
        <PlanEditor image={model.overlay} model={model} onApply={applyEdits} onClose={() => setEditorOpen(false)} />
      )}

      {siteOpen && model && (
        <SiteMap
          planW={model.bounds.x1 - model.bounds.x0}
          planD={model.bounds.y1 - model.bounds.y0}
          value={site}
          onApply={s => { setSite(s); setSiteOpen(false) }}
          onClose={() => setSiteOpen(false)}
        />
      )}

      {floorsOpen && model && (
        <FloorMarker model={model} detect={detectFloors}
          onApply={f => {
            setModel(m => m ? ({ ...m, floors: f ?? undefined } as any) : m)
            setFloorOrder((f ?? []).map((x: any) => x.id))
            setFloorNudge({})
            setStackOn(!!f && f.length > 1)
            setFloorsOpen(false)
          }}
          onClose={() => setFloorsOpen(false)} />
      )}

      {pickerOpen && pages.length > 1 && (
        <PagePicker
          pages={pages}
          current={preview?.page ?? 0}
          suggested={preview?.suggest?.page ?? null}
          onChoose={choosePage}
          onCancel={() => setPickerOpen(false)}
        />
      )}

      {cropOpen && preview && (
        <CropDialog
          image={preview.image}
          initial={crop}
          adjustments={adjustments}
          onAdjustmentsChange={setAdjustments}
          onConfirm={box => {
            setCrop(box)
            setCropOpen(false)
            setStatus(box ? 'Area selected. Click Generate.' : 'Using the full page.')
          }}
          onCancel={() => setCropOpen(false)}
        />
      )}
    </div>
  )
}