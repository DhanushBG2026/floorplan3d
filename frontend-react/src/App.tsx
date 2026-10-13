import { useState, useEffect, useMemo} from 'react'
import { FT, toFt, toM, fmtFtIn } from './units'
import Viewer from './Viewer'
import type { PlanModel } from './types'
import './App.css'
import { GROUNDS, type GroundType } from './ground'
import type { DimMode } from './types'
import { buildGroups } from './groups'
import { convertPlan, previewPlan } from './api'
import CropDialog, { type CropBox } from './CropDialog'
import ChatPanel from './ChatPanel'

interface DeleteHistory { past: number[][]; present: number[]; future: number[][] }
const EMPTY_HIST: DeleteHistory = { past: [], present: [], future: [] }

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [scale, setScale] = useState(100)
  const [knownW, setKnownW] = useState('')
  const [ppm, setPpm] = useState('')
  const [wallH, setWallH] = useState(9)
  const [dimMode, setDimMode] = useState<DimMode>('overall')
  const [model, setModel] = useState<PlanModel | null>(null)
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)
  const [ground, setGround] = useState<GroundType>('grass')
  const [heights, setHeights] = useState<Record<number, number>>({})  
  const [thicknesses, setThicknesses] = useState<Record<number, number>>({})
  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [hist, setHist] = useState<DeleteHistory>(EMPTY_HIST)
  const [preview, setPreview] = useState<{ image: string } | null>(null)
  const [crop, setCrop] = useState<CropBox | null>(null)
  const [cropOpen, setCropOpen] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const deleted = hist.present
  const groups = useMemo(() => (model ? buildGroups(model) : null), [model])
  const [docText, setDocText] = useState('')


  const onFileChosen = async (f: File | null) => {
    setFile(f)
    setPreview(null)
    setCrop(null)
    setDocText('')
    if (!f) return
    setPreviewing(true)
    setStatus('Loading preview…')
    try {
      const p = await previewPlan(f)
      setPreview(p)
      setDocText(p.text ?? '')
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
        knownWidthM: toM(parseFloat(knownW) || 0),
        pxPerM: (parseFloat(ppm) || 0) / FT,
        crop,
      })
      setModel(m)
      setHeights({})
      setThicknesses({})
      setHeights({})
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
    setThicknesses(t => {
      const n = { ...t }
      membersOf(gid).forEach(id => { delete n[id] })
      return n
    })
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

        <label>Ground</label>
        <select value={ground} onChange={e => setGround(e.target.value as GroundType)}>
          {Object.entries(GROUNDS).map(([key, g]) => (
            <option key={key} value={key}>{g.label}</option>
          ))}
        </select>

        <button onClick={generate} disabled={loading || previewing}>Generate 3D model</button>
        <div className="status">{status}</div>

        <label>Dimensions</label>
        <select value={dimMode} onChange={e => setDimMode(e.target.value as DimMode)}>
          <option value="off">Off</option>
          <option value="overall">Overall + doors/windows</option>
          <option value="all">All walls</option>
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
              
      {model && <img className="overlay" src={model.overlay} alt="detection overlay" />}
      </aside>

      <main className="view">
        {model && (
            <div className="toolbar">
              <button onClick={undo} disabled={!canUndo} title="Undo wall delete (Ctrl+Z)">↶ Undo</button>
              <button onClick={redo} disabled={!canRedo} title="Redo wall delete (Ctrl+Y)">↷ Redo</button>
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
            ground={ground}
          />
        ) : (
          <p className="empty">Upload a plan to begin</p>
        )}

        {preview && (
          <ChatPanel documentText={docText} docKey={file ? `${file.name}-${file.size}` : ''} />
        )}
        
      </main>
      {cropOpen && preview && (
        <CropDialog
          image={preview.image}
          initial={crop}
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