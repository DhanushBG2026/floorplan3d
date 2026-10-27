// Pick where the model stands: search or zoom to the plot, draw its boundary, then place the
// house footprint inside it (drag it, rotate it). Needs:  npm i leaflet  &&  npm i -D @types/leaflet
import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import {
  ATTRIBUTION, TILE_URL, footprint, insidePolygon, polygonAreaM2, type SiteConfig,
} from './site'

interface Props {
  planW: number                   // plan width and depth in metres
  planD: number
  value: SiteConfig | null
  onApply: (s: SiteConfig | null) => void
  onClose: () => void
}
type Mode = 'boundary' | 'house'
type LL = [number, number]

export default function SiteMap({ planW, planD, value, onApply, onClose }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const tiles = useRef<L.TileLayer | null>(null)
  const draw = useRef<L.LayerGroup | null>(null)
  const dragging = useRef(false)

  const [layer, setLayer] = useState<'satellite' | 'street'>(value?.layer ?? 'satellite')
  const [mode, setMode] = useState<Mode>(value ? 'house' : 'boundary')
  const [boundary, setBoundary] = useState<LL[]>(value?.boundary ?? [])
  const [centre, setCentre] = useState<LL | null>(value ? [value.lat, value.lng] : null)
  const [rot, setRot] = useState(value?.rotationDeg ?? 0)
  const [zoom, setZoom] = useState(value?.zoom ?? 18)
  const [q, setQ] = useState('')
  const [msg, setMsg] = useState('')

  const modeRef = useRef(mode); modeRef.current = mode
  const centreRef = useRef(centre); centreRef.current = centre

  // ---- create the map once ----
  useEffect(() => {
    const m = L.map(box.current!, { maxZoom: 21, zoomControl: true }).setView(value ? [value.lat, value.lng] : [20.5937, 78.9629], value ? 19 : 5)
    map.current = m
    draw.current = L.layerGroup().addTo(m)
    m.on('click', (e: L.LeafletMouseEvent) => {
      if (modeRef.current === 'boundary') setBoundary(b => [...b, [e.latlng.lat, e.latlng.lng]])
      else setCentre([e.latlng.lat, e.latlng.lng])
    })
    m.on('mousemove', (e: L.LeafletMouseEvent) => { if (dragging.current) setCentre([e.latlng.lat, e.latlng.lng]) })
    m.on('mouseup', () => { if (dragging.current) { dragging.current = false; m.dragging.enable() } })
    setTimeout(() => m.invalidateSize(), 50)
    return () => { m.remove(); map.current = null }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- base layer ----
  useEffect(() => {
    const m = map.current
    if (!m) return
    tiles.current?.remove()
    tiles.current = L.tileLayer(TILE_URL[layer], { maxNativeZoom: 19, maxZoom: 21, attribution: ATTRIBUTION[layer] }).addTo(m)
  }, [layer])

  const corners = useMemo(() => centre ? footprint(centre[0], centre[1], planW, planD, rot) : null, [centre, planW, planD, rot])
  const outside = !!corners && boundary.length >= 3 && corners.some(c => !insidePolygon(c, boundary))
  const area = polygonAreaM2(boundary)

  // ---- drawings: boundary and footprint ----
  useEffect(() => {
    const g = draw.current, m = map.current
    if (!g || !m) return
    g.clearLayers()
    if (boundary.length) {
      L.polygon(boundary, { color: '#ffd400', weight: 3, fillOpacity: 0.08, interactive: false }).addTo(g)
      boundary.forEach(p => L.circleMarker(p, { radius: 4, color: '#ffd400', fillOpacity: 1, interactive: false }).addTo(g))
    }
    if (corners) {
      const poly = L.polygon(corners, { color: outside ? '#ff3b30' : '#00e0ff', weight: 2, fillOpacity: 0.35, bubblingMouseEvents: false }).addTo(g)
      poly.on('mousedown', () => { dragging.current = true; m.dragging.disable() })
      // a short line from the middle to the top of the plan, so the turn is visible
      const mid: LL = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2]
      L.polyline([centre!, mid], { color: '#fff', weight: 2, dashArray: '4 4', interactive: false }).addTo(g)
    }
  }, [boundary, corners, outside, centre])

  const search = async () => {
    if (!q.trim()) return
    setMsg('Searching…')
    try {
      const r = await fetch(`https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`)
      const j = await r.json()
      if (!j.length) { setMsg('Nothing found'); return }
      map.current?.setView([Number(j[0].lat), Number(j[0].lon)], 18)
      setMsg('')
    } catch { setMsg('Search failed (offline?)') }
  }
  const locate = () => navigator.geolocation?.getCurrentPosition(
    p => map.current?.setView([p.coords.latitude, p.coords.longitude], 18),
    () => setMsg('Location not allowed'))
  const centreOnPlot = () => {
    if (!boundary.length) return
    setCentre([boundary.reduce((s, p) => s + p[0], 0) / boundary.length, boundary.reduce((s, p) => s + p[1], 0) / boundary.length])
    setMode('house')
  }
  const apply = () => {
    if (!centre) return
    onApply({ lat: centre[0], lng: centre[1], rotationDeg: rot, boundary, layer, zoom })
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 60, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ background: '#fff', color: '#111', borderRadius: 10, width: 'min(1200px, 96vw)', height: '92vh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', padding: '8px 12px', borderBottom: '1px solid #ddd' }}>
          <strong>Place on site</strong>
          <input placeholder="Search address or place" value={q} style={{ width: 220 }}
                 onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === 'Enter' && search()} />
          <button onClick={search}>Search</button>
          <button onClick={locate}>My location</button>
          <select value={layer} onChange={e => setLayer(e.target.value as 'satellite' | 'street')}>
            <option value="satellite">Satellite</option>
            <option value="street">Street map</option>
          </select>
          <span style={{ color: '#b00' }}>{msg}</span>
          <span style={{ flex: 1 }} />
          {value && <button onClick={() => onApply(null)}>Remove site</button>}
          <button onClick={onClose}>Cancel</button>
          <button onClick={apply} disabled={!centre} style={{ fontWeight: 700 }}>Place model here</button>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center', padding: '6px 12px', fontSize: 13 }}>
          <button onClick={() => setMode('boundary')} style={{ fontWeight: mode === 'boundary' ? 700 : 400, outline: mode === 'boundary' ? '2px solid #b8a000' : 'none' }}>1. Draw plot boundary</button>
          <button onClick={() => setMode('house')} style={{ fontWeight: mode === 'house' ? 700 : 400, outline: mode === 'house' ? '2px solid #0099b3' : 'none' }}>2. Place house</button>
          <button onClick={() => setBoundary(b => b.slice(0, -1))} disabled={!boundary.length}>Undo point</button>
          <button onClick={() => setBoundary([])} disabled={!boundary.length}>Clear boundary</button>
          <button onClick={centreOnPlot} disabled={boundary.length < 3}>Centre house in plot</button>
          <label>Turn <input type="range" min="0" max="359" value={rot} onChange={e => setRot(Number(e.target.value))} /> {rot}°</label>
          <label>3D detail <select value={zoom} onChange={e => setZoom(Number(e.target.value))}>
            <option value={18}>Normal</option><option value={19}>Sharp</option></select></label>
          <span style={{ color: '#555' }}>
            {mode === 'boundary' ? 'Click the plot corners one by one.' : 'Click to set the house position, or drag the cyan footprint.'}
            {area > 0 && ` Plot ≈ ${Math.round(area)} m² (${Math.round(area * 10.7639)} sq ft).`}
          </span>
          {outside && <span style={{ color: '#d00', fontWeight: 600 }}>House goes outside the plot</span>}
        </div>
        <div ref={box} style={{ flex: 1, cursor: mode === 'boundary' ? 'crosshair' : 'pointer' }} />
      </div>
    </div>
  )
}