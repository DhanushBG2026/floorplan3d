import { Component, Suspense, createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import {
  OrbitControls, Html, Line, Edges, useTexture, Environment, Lightformer, Sky,
} from '@react-three/drei'
import type { PlanModel, Opening, DimMode, Railing, Stair, Wall } from './types'
import type { WallGroups } from './groups'
import { GROUNDS, type GroundType } from './ground'
import { fmtFtIn } from './units'

type DimensionUnit = 'ft-in' | 'm' | 'cm' | 'mm' | 'in'
import { roofTriangles, planarUVs, type RoofType } from './roof'
import { WALL_FINISHES, ROOF_FINISHES, DEFAULT_ROOF_COLOR, type Finish } from './materials'
import { applyExtensions, groupAxis, snapCoordinate, clampExt, type Ext, type AxisInfo } from './extend'

export interface RoofConfig { type: RoofType; visible: boolean; ridgeM: number; overhangM: number }
export interface OpeningSpec { sillM: number; headM: number; doorM: number }
interface Box2 { x0: number; z0: number; x1: number; z1: number }

const DOOR_WOOD = '#8a5a32'
const DOOR_FRAME = '#5c3b1e'

const DOOR_H = 7 * 0.3048   // every door is 7 ft high
const SILL = 0.9
const WIN_TOP = 2.1
const WALL = '#e8e4dc'

type V3 = [number, number, number]

// ---------- Box (walls, floor, window parts) ----------
// Plan coords: x right, y down (m) -> three: x right, z = plan y, y up
interface BoxProps {
  x0: number; z0: number; x1: number; z1: number
  y0: number; y1: number
  color: string
  opacity?: number
  shrink?: number
  edges?: boolean
  glossy?: boolean
  selected?: boolean
  onClick?: (e: ThreeEvent<MouseEvent>) => void
}

export interface FinishConfig { wall: Finish; roof: Finish; roofColor: string }

const FinishContext = createContext<FinishConfig>({
  wall: WALL_FINISHES.plain,
  roof: ROOF_FINISHES.plain,
  roofColor: DEFAULT_ROOF_COLOR,
})

// If a texture file is missing, show a plain colour instead of crashing the whole viewer.
class TexBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch(e: unknown) { console.warn('Texture failed to load, using a plain colour instead.', e) }
  render() { return this.state.failed ? this.props.fallback : this.props.children }
}

interface SurfaceProps { finish: Finish; tint?: string; selected?: boolean; side?: THREE.Side }

function PlainSurface({ finish, tint, selected, side }: SurfaceProps) {
  return (
    <meshPhysicalMaterial
      color={tint ?? finish.color}
      roughness={finish.roughness}
      metalness={finish.metalness}
      clearcoat={finish.map ? 0 : 1}
      clearcoatRoughness={0.1}
      emissive={selected ? '#3b82f6' : '#000000'}
      emissiveIntensity={selected ? 0.4 : 0}
      side={side}
    />
  )
}

function TexturedSurface({ finish, tint, selected, side }: SurfaceProps) {
  const maps = useTexture({
    map: finish.map as string,
    ...(finish.normal ? { normalMap: finish.normal } : {}),
  }) as { map: THREE.Texture; normalMap?: THREE.Texture }

  useEffect(() => {
    maps.map.colorSpace = THREE.SRGBColorSpace
    for (const t of [maps.map, maps.normalMap]) {
      if (!t) continue
      t.wrapS = t.wrapT = THREE.RepeatWrapping
      t.anisotropy = 8
      t.needsUpdate = true
    }
  }, [maps.map, maps.normalMap])

  return (
    <meshPhysicalMaterial
      map={maps.map}
      normalMap={maps.normalMap}
      color={tint ?? '#ffffff'}
      roughness={finish.roughness}
      metalness={finish.metalness}
      emissive={selected ? '#3b82f6' : '#000000'}
      emissiveIntensity={selected ? 0.4 : 0}
      side={side}
    />
  )
}

function Surface(props: SurfaceProps) {
  const plain = <PlainSurface {...props} />
  if (!props.finish.map) return plain
  return (
    <TexBoundary fallback={plain}>
      <Suspense fallback={plain}>
        <TexturedSurface {...props} />
      </Suspense>
    </TexBoundary>
  )
}

// A box whose texture coordinates are in metres, so patterns keep their real size on every wall.
function scaledBoxGeometry(w: number, h: number, d: number, tile: number) {
  const g = new THREE.BoxGeometry(w, h, d)
  const uv = g.attributes.uv as THREE.BufferAttribute
  const size: [number, number][] = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]]
  for (let f = 0; f < 6; f++) {
    for (let i = 0; i < 4; i++) {
      const k = f * 4 + i
      uv.setXY(k, (uv.getX(k) * size[f][0]) / tile, (uv.getY(k) * size[f][1]) / tile)
    }
  }
  uv.needsUpdate = true
  return g
}

function Box({
  x0, z0, x1, z1, y0, y1, color, opacity = 1, shrink = 0, edges, glossy, selected, onClick,
}: BoxProps) {
  const { wall } = useContext(FinishContext)
  const w = x1 - x0
  const d = z1 - z0
  const h = y1 - y0 - shrink
  const tile = glossy ? wall.tileM : 1
  const valid = w > 0 && d > 0 && h > 0
  const geometry = useMemo(() => (valid ? scaledBoxGeometry(w, h, d, tile) : null), [valid, w, h, d, tile])
  useEffect(() => () => geometry?.dispose(), [geometry])
  if (!geometry) return null

  return (
    <mesh
      position={[(x0 + x1) / 2, y0 + h / 2, (z0 + z1) / 2]}
      castShadow={opacity === 1}
      receiveShadow
      onClick={onClick}
      onPointerOver={onClick ? e => { e.stopPropagation(); document.body.style.cursor = 'pointer' } : undefined}
      onPointerOut={onClick ? () => { document.body.style.cursor = 'auto' } : undefined}
    >
      <primitive object={geometry} attach="geometry" />
      {glossy ? (
        <Surface finish={wall} selected={selected} />
      ) : (
        <meshStandardMaterial color={color} transparent={opacity < 1} opacity={opacity} />
      )}
      {edges && <Edges color={selected ? '#3b82f6' : '#777777'} />}
    </mesh>
  )
}

function DoorParts({ o, h, doorH }: { o: Opening; h: number; doorH: number }) {
  const t = o.thickness
  const c = o.centre
  const dh = Math.min(doorH, h)
  const f = 0.05
  const leafT = 0.045
  const knobY = Math.min(1.0, dh / 2)
  const knobAlong = o.end - f - 0.12

  const part = (
    a0: number, a1: number, p0: number, p1: number,
    y0: number, y1: number, color: string, glossy = false,
  ) =>
    o.horizontal ? (
      <Box x0={a0} x1={a1} z0={p0} z1={p1} y0={y0} y1={y1} color={color} glossy={glossy} />
    ) : (
      <Box x0={p0} x1={p1} z0={a0} z1={a1} y0={y0} y1={y1} color={color} glossy={glossy} />
    )

  const knob = (side: 1 | -1) => {
    const p = c + side * (leafT / 2 + 0.02)
    const pos: V3 = o.horizontal ? [knobAlong, knobY, p] : [p, knobY, knobAlong]
    return (
      <mesh position={pos} castShadow>
        <sphereGeometry args={[0.03, 16, 16]} />
        <meshStandardMaterial color="#d4af37" metalness={0.8} roughness={0.3} />
      </mesh>
    )
  }

  return (
    <>
      {part(o.start, o.start + f, c - t / 2, c + t / 2, 0, dh, DOOR_FRAME)}
      {part(o.end - f, o.end, c - t / 2, c + t / 2, 0, dh, DOOR_FRAME)}
      {part(o.start + f, o.end - f, c - leafT / 2, c + leafT / 2, 0.01, dh - 0.01, DOOR_WOOD)}
      {knob(1)}
      {knob(-1)}
    </>
  )
}

interface OpeningMeshProps {
  o: Opening
  H: number
  spec: OpeningSpec
  selected?: boolean
  onPick?: (e: ThreeEvent<MouseEvent>) => void
}

function RailingMesh({ r, height = 1.0 }: { r: Railing; height?: number }) {
  const cx = (r.x0 + r.x1) / 2
  const cz = (r.y0 + r.y1) / 2
  const length = r.horizontal ? r.x1 - r.x0 : r.y1 - r.y0
  const postCount = Math.max(2, Math.ceil(length / 0.9) + 1)
  const postRadius = 0.035
  const railRadius = 0.035

  const posts = Array.from({ length: postCount }, (_, i) => {
    const t = i / (postCount - 1)
    return r.horizontal
      ? { x: r.x0 + t * length, z: cz }
      : { x: cx, z: r.y0 + t * length }
  })

  return (
    <group>
      {posts.map((p, i) => (
        <mesh key={`post-${r.id}-${i}`} position={[p.x, height / 2, p.z]} castShadow>
          <cylinderGeometry args={[postRadius, postRadius, height, 8]} />
          <meshStandardMaterial color="#444b54" metalness={0.65} roughness={0.32} />
        </mesh>
      ))}
      <mesh
        position={[cx, height - railRadius, cz]}
        // a cylinder stands along Y by default: lay it along X (horizontal) or Z (vertical in plan)
        rotation-z={r.horizontal ? Math.PI / 2 : 0}
        rotation-x={r.horizontal ? 0 : Math.PI / 2}
        castShadow
      >
        <cylinderGeometry args={[railRadius, railRadius, Math.max(0.05, length), 8]} />
        <meshStandardMaterial color="#303740" metalness={0.7} roughness={0.28} />
      </mesh>
    </group>
  )
}

// ---------- NEW: Staircase renderer ----------
function StairsMesh({ stair, height = 1.2 }: { stair: Stair; height?: number }) {
  const count = Math.max(3, Math.min(16, stair.steps || 8))
  const run = stair.length / count

  return (
    <group>
      {Array.from({ length: count }, (_, i) => {
        const stepHeight = height * (i + 1) / count

        // For horizontal stairs, each step advances along X.
        // For vertical stairs, each step advances along the plan's Y/Z axis.
        const x0 = stair.horizontal ? stair.x0 + i * run : stair.x0
        const x1 = stair.horizontal ? stair.x0 + (i + 1) * run : stair.x1
        const z0 = stair.horizontal ? stair.y0 : stair.y0 + i * run
        const z1 = stair.horizontal ? stair.y1 : stair.y0 + (i + 1) * run

        return (
          <Box
            key={`stair-${stair.id}-${i}`}
            x0={x0}
            x1={x1}
            z0={z0}
            z1={z1}
            y0={0}
            y1={stepHeight}
            color="#b8b1a5"
          />
        )
      })}
    </group>
  )
}

function OpeningMesh({ o, H, spec, selected, onPick }: OpeningMeshProps) {
  const t = o.thickness

  const r = (y0: number, y1: number, color: string, opacity = 1) => {
    const isWall = color === WALL
    const common = {
      y0, y1, color, opacity,
      glossy: isWall,
      selected: !!selected && isWall,
      onClick: isWall ? onPick : undefined,
    }

    return o.horizontal ? (
      <Box x0={o.start} z0={o.centre - t / 2} x1={o.end} z1={o.centre + t / 2} {...common} />
    ) : (
      <Box x0={o.centre - t / 2} z0={o.start} x1={o.centre + t / 2} z1={o.end} {...common} />
    )
  }

  const top = Math.min(o.type === 'window' ? spec.headM : DOOR_H, H)
  const sill = Math.min(spec.sillM, top)

  return (
    <>
      {r(top, H, WALL)}
      {o.type === 'window' && (
        <>
          {r(0, sill, WALL)}
          {r(sill, top, '#9ad1ff', 0.35)}
        </>
      )}
      {o.type === 'door' && <DoorParts o={o} h={H} doorH={DOOR_H} />}
    </>
  )
}

// ---------- Dimensions ----------
type DimKind = 'overall' | 'wall' | 'window' | 'door' | 'opening'

const DIM_COLORS: Record<DimKind, string> = {
  overall: '#ffffff',
  wall: '#9ca3af',
  window: '#38bdf8',
  door: '#4ade80',
  opening: '#c084fc',
}

interface DimProps {
  a: V3
  b: V3
  offset?: V3
  text: string
  kind: DimKind
  scale: number
}

function Dim({ a, b, offset = [0, 0, 0], text, kind, scale }: DimProps) {
  const p0: V3 = [a[0] + offset[0], a[1] + offset[1], a[2] + offset[2]]
  const p1: V3 = [b[0] + offset[0], b[1] + offset[1], b[2] + offset[2]]
  const mid: V3 = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2]

  return (
    <>
      <Line points={[a, p0, p1, b]} color={DIM_COLORS[kind]} lineWidth={kind === 'overall' ? 1.5 : 1} />
      <Html
        position={mid}
        center
        distanceFactor={scale}
        zIndexRange={[10, 0]}
        style={{ pointerEvents: 'none' }}
      >
        <div className={`dim ${kind}`}>{text}</div>
      </Html>
    </>
  )
}

function formatDimension(metres: number, unit: DimensionUnit): string {
  switch (unit) {
    case 'm':
      return `${metres.toFixed(2)} m`
    case 'cm':
      return `${(metres * 100).toFixed(1)} cm`
    case 'mm':
      return `${Math.round(metres * 1000)} mm`
    case 'in':
      return `${(metres / 0.0254).toFixed(1)} in`
    case 'ft-in':
    default:
      return fmtFtIn(metres)
  }
}

function Dimensions({
  model, H, mode, scale, unit,
}: { model: PlanModel; H: number; mode: DimMode; scale: number; unit: DimensionUnit }) {
  if (mode === 'off') return null

  const b = model.bounds
  const cx = (b.x0 + b.x1) / 2
  const cz = (b.y0 + b.y1) / 2
  const gap = 1.2
  const fmt = (v: number) => formatDimension(v, unit)

  return (
    <>
      {/* Overall width and depth */}
      <Dim kind="overall" scale={scale} a={[b.x0, 0.03, b.y1]} b={[b.x1, 0.03, b.y1]}
        offset={[0, 0, gap]} text={fmt(b.x1 - b.x0)} />
      <Dim kind="overall" scale={scale} a={[b.x0, 0.03, b.y0]} b={[b.x0, 0.03, b.y1]}
        offset={[-gap, 0, 0]} text={fmt(b.y1 - b.y0)} />

      {/* Individual wall dimensions */}
      {mode === 'all' &&
        model.walls.filter(w => w.length >= 1.5 && !w.diagonal).map(w => {
          const hz = w.horizontal
          const mid = hz ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
          const outward = hz ? (mid < cz ? -1 : 1) : (mid < cx ? -1 : 1)
          const off = (w.thickness / 2 + 0.35) * outward
          const a: V3 = hz ? [w.x0, H + 0.02, mid] : [mid, H + 0.02, w.y0]
          const c: V3 = hz ? [w.x1, H + 0.02, mid] : [mid, H + 0.02, w.y1]
          const offset: V3 = hz ? [0, 0, off] : [off, 0, 0]
          return <Dim key={`w${w.id}`} kind="wall" scale={scale} a={a} b={c} offset={offset} text={fmt(w.length)} />
        })}

      {/* Opening dimensions */}
      {model.openings.map(o => {
        const y = H + 0.12
        const a: V3 = o.horizontal ? [o.start, y, o.centre] : [o.centre, y, o.start]
        const c: V3 = o.horizontal ? [o.end, y, o.centre] : [o.centre, y, o.end]
        return <Dim key={`o${o.id}`} kind={o.type} scale={scale} a={a} b={c} text={fmt(o.width)} />
      })}
    </>
  )
}

// ---------- Ground ----------
function Ground({ type, cx, cz, size }: { type: GroundType; cx: number; cz: number; size: number }) {
  const { texture: url, tileM } = GROUNDS[type]
  const extent = Math.min(Math.max(size * 12, 120), 800)
  const base = useTexture(url)
  const maxAniso = useThree(s => s.gl.capabilities.getMaxAnisotropy())

  const tex = useMemo(() => {
    const t = base.clone()
    t.wrapS = t.wrapT = THREE.RepeatWrapping
    t.repeat.set(extent / tileM, extent / tileM)
    t.colorSpace = THREE.SRGBColorSpace
    t.anisotropy = maxAniso
    t.needsUpdate = true
    return t
  }, [base, extent, tileM, maxAniso])

  return (
    <mesh position={[cx, -0.11, cz]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
      <planeGeometry args={[extent, extent]} />
      <meshStandardMaterial map={tex} roughness={1} />
    </mesh>
  )
}

// ---------- Lights ----------
function Lights({ cx, cz, size }: { cx: number; cz: number; size: number }) {
  const ref = useRef<THREE.DirectionalLight>(null)

  useEffect(() => {
    const l = ref.current
    if (!l) return
    l.target.position.set(cx, 0, cz)
    l.target.updateMatrixWorld()
  }, [cx, cz])

  const d = size * 0.9

  return (
    <>
      <hemisphereLight args={['#ffffff', '#444455', 0.6]} />
      <directionalLight
        ref={ref}
        position={[cx + size * 0.7, size * 1.3, cz + size * 0.5]}
        intensity={2.2}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-bias={-0.0005}
        shadow-normalBias={0.03}
      >
        <orthographicCamera attach="shadow-camera" args={[-d, d, d, -d, 0.1, size * 5]} />
      </directionalLight>
    </>
  )
}

function Roof({ cfg, box, eaveY }: { cfg: RoofConfig; box: Box2; eaveY: number }) {
  const { wall, roof: roofFinish, roofColor } = useContext(FinishContext)

  const geo = useMemo(() => {
    if (cfg.type === 'none') return null

    const t = roofTriangles({ type: cfg.type, ...box, eaveY, ridgeY: cfg.ridgeM, overhang: cfg.overhangM })
    if (!t.roof.length) return null

    const make = (pos: number[], tile: number, swap: boolean) => {
      const g = new THREE.BufferGeometry()
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
      g.setAttribute('uv', new THREE.Float32BufferAttribute(planarUVs(pos, tile, swap), 2))
      g.computeVertexNormals()
      return g
    }

    return {
      roof: make(t.roof, roofFinish.tileM, !!roofFinish.swapUV),
      gables: t.gables.length ? make(t.gables, wall.tileM, false) : null,
    }
  }, [cfg.type, cfg.ridgeM, cfg.overhangM, box.x0, box.z0, box.x1, box.z1, eaveY,
      roofFinish.tileM, roofFinish.swapUV, wall.tileM])

  if (!geo || !cfg.visible) return null

  return (
    <>
      <mesh geometry={geo.roof} castShadow receiveShadow>
        <Surface finish={roofFinish} tint={roofColor} side={THREE.DoubleSide} />
      </mesh>
      {geo.gables && (
        <mesh geometry={geo.gables} castShadow receiveShadow>
          <Surface finish={wall} side={THREE.DoubleSide} />
        </mesh>
      )}
    </>
  )
}

function Stumps({ box, height }: { box: Box2; height: number }) {
  if (height < 0.15) return null

  const inset = 0.25
  const w = box.x1 - box.x0
  const d = box.z1 - box.z0
  const nx = Math.max(2, Math.ceil(w / 2.4) + 1)
  const nz = Math.max(2, Math.ceil(d / 2.4) + 1)
  const posts: [number, number][] = []

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < nz; j++) {
      posts.push([
        box.x0 + inset + ((w - 2 * inset) * i) / (nx - 1),
        box.z0 + inset + ((d - 2 * inset) * j) / (nz - 1),
      ])
    }
  }

  const h = height - 0.1

  return (
    <>
      {posts.map(([x, z], i) => (
        <mesh key={i} position={[x, h / 2, z]} castShadow receiveShadow>
          <boxGeometry args={[0.15, h, 0.15]} />
          <meshStandardMaterial color="#9a9a94" roughness={0.9} />
        </mesh>
      ))}
    </>
  )
}

const NO_EXT: Record<number, Ext> = {}

// Two draggable dots at the ends of the selected wall. Dragging moves that end along the wall.
function ExtendHandles({
  gid, axis, ext, wallH, stumpM, snapOn, walls, skip, onChange,
}: {
  gid: number
  axis: AxisInfo
  ext: Ext
  wallH: number
  stumpM: number
  snapOn: boolean
  walls: Wall[]
  skip: number[]
  onChange: (gid: number, e: Ext) => void
}) {
  const { camera, gl, raycaster, controls } = useThree()
  const [drag, setDrag] = useState<'start' | 'end' | null>(null)
  const [snapped, setSnapped] = useState(false)
  const latest = useRef({ ext, walls, skip, snapOn, axis, onChange })
  latest.current = { ext, walls, skip, snapOn, axis, onChange }
  const y = wallH + 0.3

  useEffect(() => {
    if (!drag) return
    const ctl = controls as unknown as { enabled: boolean } | null
    if (ctl) ctl.enabled = false // no camera orbit while dragging a handle

    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(stumpM + y))
    const hit = new THREE.Vector3()
    const ndc = new THREE.Vector2()

    const move = (ev: PointerEvent) => {
      const r = gl.domElement.getBoundingClientRect()
      ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1)
      raycaster.setFromCamera(ndc, camera)
      if (!raycaster.ray.intersectPlane(plane, hit)) return

      const { ext: cur, walls: ws, skip: sk, snapOn: sn, axis: ax, onChange: change } = latest.current
      let next: Ext
      let didSnap = false

      if (ax.diagonal) {
        const L = ax.hiBase
        const dx = ((ax.bx as number) - (ax.ax as number)) / L
        const dz = ((ax.by as number) - (ax.ay as number)) / L
        const t = (hit.x - (ax.ax as number)) * dx + (hit.z - (ax.ay as number)) * dz
        next = drag === 'start' ? { start: -t, end: cur.end } : { start: cur.start, end: t - L }
      } else {
        let v = ax.horizontal ? hit.x : hit.z
        if (sn) {
          const s = snapCoordinate({
            walls: ws, skip: sk, horizontal: ax.horizontal, perp: ax.perp, value: v, threshold: 0.3,
          })
          v = s.value
          didSnap = s.targetId !== null
        }
        next = drag === 'start'
          ? { start: ax.loBase - v, end: cur.end }
          : { start: cur.start, end: v - ax.hiBase }
      }
      setSnapped(didSnap)
      change(gid, clampExt(ax, next))
    }
    const up = () => { setDrag(null); setSnapped(false) }

    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
    return () => {
      window.removeEventListener('pointermove', move)
      window.removeEventListener('pointerup', up)
      if (ctl) ctl.enabled = true
    }
  }, [drag, gid, stumpM, y, camera, gl, raycaster, controls])

  const point = (which: 'start' | 'end'): V3 => {
    if (axis.diagonal) {
      const L = axis.hiBase
      const dx = ((axis.bx as number) - (axis.ax as number)) / L
      const dz = ((axis.by as number) - (axis.ay as number)) / L
      const t = which === 'start' ? -ext.start : L + ext.end
      return [(axis.ax as number) + dx * t, y, (axis.ay as number) + dz * t]
    }
    const v = which === 'start' ? axis.loBase - ext.start : axis.hiBase + ext.end
    return axis.horizontal ? [v, y, axis.perp] : [axis.perp, y, v]
  }

  const handle = (which: 'start' | 'end') => {
    const active = drag === which
    return (
      <mesh
        key={which}
        position={point(which)}
        renderOrder={999}
        onPointerDown={e => { e.stopPropagation(); setDrag(which) }}
        onPointerOver={e => { e.stopPropagation(); document.body.style.cursor = 'grab' }}
        onPointerOut={() => { document.body.style.cursor = 'auto' }}
      >
        <sphereGeometry args={[active ? 0.18 : 0.14, 20, 20]} />
        <meshBasicMaterial
          color={active && snapped ? '#22c55e' : '#f59e0b'}
          depthTest={false}
          transparent
          opacity={0.95}
        />
      </mesh>
    )
  }

  return <>{handle('start')}{handle('end')}</>
}

// Split a wall's plan-length at detected openings that lie on the same wall run.
// OpeningMesh supplies the lintel/sill/glazing, so the full-height wall must not
// occupy the opening interval underneath it.
function wallLengthSegments(w: Wall, openings: Opening[]) {
  const horizontal = !w.diagonal && w.horizontal
  const start = horizontal ? w.x0 : w.y0
  const end = horizontal ? w.x1 : w.y1
  const centre = horizontal ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
  const thickness = Number(w.thickness) || 0.12
  const cuts = openings
    .filter(o => o.horizontal === horizontal)
    .filter(o => Math.abs(o.centre - centre) <= Math.max(thickness, o.thickness || 0, 0.12))
    .map(o => [Math.max(start, o.start), Math.min(end, o.end)] as const)
    .filter(([a, b]) => b - a > 0.015)
    .sort((a, b) => a[0] - b[0])

  const merged: [number, number][] = []
  for (const cut of cuts) {
    const last = merged[merged.length - 1]
    if (last && cut[0] <= last[1]) last[1] = Math.max(last[1], cut[1])
    else merged.push([cut[0], cut[1]])
  }
  const segments: [number, number][] = []
  let cursor = start
  for (const [a, b] of merged) {
    if (a - cursor > 0.015) segments.push([cursor, a])
    cursor = Math.max(cursor, b)
  }
  if (end - cursor > 0.015) segments.push([cursor, end])
  return segments
}

// ---------- Design read from a PDF: roof + deck ----------
// `model.design` is produced by the backend (pdf_context.py) when a PDF with elevations is uploaded.
export interface DesignInfo {
  levels?: { ffl?: number; doorTop?: number; ceiling?: number; wallTop?: number; ridge?: number }
  roof?: { type: string; riseM: number; ridgeLengthM: number | null; verandah: boolean } | null
  deck?: { side: 'top' | 'bottom' | 'left' | 'right'; dimsMm: [number, number]; roofed: boolean } | null
}

type P3 = [number, number, number]
const TIMBER = '#8b6a47'
const STEEL = '#2c3036'

// A thin roof slab: top surface = 4 corners, thickness `t` below it.
function slabGeometry(c: [P3, P3, P3, P3], t: number, tile: number, swap: boolean) {
  const lo = c.map(p => [p[0], p[1] - t, p[2]] as P3)
  const pos: number[] = []
  const tri = (a: P3, b: P3, d: P3) => pos.push(...a, ...b, ...d)
  const quad = (a: P3, b: P3, d: P3, e: P3) => { tri(a, b, d); tri(a, d, e) }
  quad(c[0], c[1], c[2], c[3])
  quad(lo[3], lo[2], lo[1], lo[0])
  for (let i = 0; i < 4; i++) quad(c[i], c[(i + 1) % 4], lo[(i + 1) % 4], lo[i])
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(planarUVs(pos, tile, swap), 2))
  g.computeVertexNormals()
  return g
}

function triGeometry(a: P3, b: P3, c: P3, tile: number) {
  const pos = [...a, ...b, ...c]
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
  g.setAttribute('uv', new THREE.Float32BufferAttribute(planarUVs(pos, tile, false), 2))
  g.computeVertexNormals()
  return g
}

interface DeckRect extends Box2 { side: 'top' | 'bottom' | 'left' | 'right'; along: number; depth: number }

// Gable roof from the PDF: ridge along `axis`, rise above the wall top, overhang all round,
// plus the gable-end infill triangles.
function DesignRoof({ box, eaveY, riseM, axis, overhang }: {
  box: Box2; eaveY: number; riseM: number; axis: 'x' | 'z'; overhang: number
}) {
  const { wall, roof: roofFinish, roofColor } = useContext(FinishContext)
  const geo = useMemo(() => {
    const u0 = axis === 'x' ? box.x0 : box.z0, u1 = axis === 'x' ? box.x1 : box.z1   // along the ridge
    const v0 = axis === 'x' ? box.z0 : box.x0, v1 = axis === 'x' ? box.z1 : box.x1   // across it
    const vc = (v0 + v1) / 2
    const ov = Math.max(0, overhang)
    const rise = Math.max(0.05, riseM)
    const slope = rise / Math.max(0.1, (v1 - v0) / 2)
    const drop = ov * slope
    const ridgeY = eaveY + rise
    const at = (u: number, y: number, v: number): P3 => (axis === 'x' ? [u, y, v] : [v, y, u])
    const t = 0.09
    return {
      a: slabGeometry([at(u0 - ov, eaveY - drop, v0 - ov), at(u1 + ov, eaveY - drop, v0 - ov),
                       at(u1 + ov, ridgeY, vc), at(u0 - ov, ridgeY, vc)], t, roofFinish.tileM, !!roofFinish.swapUV),
      b: slabGeometry([at(u0 - ov, ridgeY, vc), at(u1 + ov, ridgeY, vc),
                       at(u1 + ov, eaveY - drop, v1 + ov), at(u0 - ov, eaveY - drop, v1 + ov)], t, roofFinish.tileM, !!roofFinish.swapUV),
      g0: triGeometry(at(u0, eaveY, v0), at(u0, eaveY, v1), at(u0, ridgeY - 0.001, vc), wall.tileM),
      g1: triGeometry(at(u1, eaveY, v1), at(u1, eaveY, v0), at(u1, ridgeY - 0.001, vc), wall.tileM),
    }
  }, [box.x0, box.z0, box.x1, box.z1, eaveY, riseM, axis, overhang, roofFinish.tileM, roofFinish.swapUV, wall.tileM])

  return (
    <>
      {[geo.a, geo.b].map((g, i) => (
        <mesh key={i} geometry={g} castShadow receiveShadow>
          <Surface finish={roofFinish} tint={roofColor} side={THREE.DoubleSide} />
        </mesh>
      ))}
      {[geo.g0, geo.g1].map((g, i) => (
        <mesh key={`g${i}`} geometry={g} castShadow receiveShadow>
          <Surface finish={wall} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  )
}

function boardTexture(vertical: boolean, w: number, d: number) {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')!
  const shades = ['#8d6c49', '#84643f', '#917150', '#8a6a46']
  for (let i = 0; i < 8; i++) {
    g.fillStyle = shades[i % shades.length]
    if (vertical) g.fillRect(i * 16, 0, 16, 128)
    else g.fillRect(0, i * 16, 128, 16)
    g.fillStyle = 'rgba(0,0,0,0.35)'
    if (vertical) g.fillRect(i * 16, 0, 1.5, 128)
    else g.fillRect(0, i * 16, 128, 1.5)
  }
  const t = new THREE.CanvasTexture(c)
  t.wrapS = t.wrapT = THREE.RepeatWrapping
  t.colorSpace = THREE.SRGBColorSpace
  // 8 boards of 0.14 m = 1.12 m per tile across the boards, one tile along them
  t.repeat.set(vertical ? w / 1.12 : 1, vertical ? 1 : d / 1.12)
  return t
}

// Deck platform with board lines, balustrade (gaps where stairs meet it), posts and verandah roof.
function Deck({ rect, stairs, eaveY, roofed }: {
  rect: DeckRect; stairs: Stair[]; eaveY: number; roofed: boolean
}) {
  const { roof: roofFinish, roofColor } = useContext(FinishContext)
  const w = rect.x1 - rect.x0
  const d = rect.z1 - rect.z0
  const vertical = rect.side === 'left' || rect.side === 'right'           // boards run along the wall
  const tex = useMemo(() => boardTexture(vertical, w, d), [vertical, w, d])
  useEffect(() => () => tex.dispose(), [tex])

  // outer edges (not the one against the building): [ax, az, bx, bz]
  const edges: [number, number, number, number][] = []
  const { x0, z0, x1, z1 } = rect
  if (rect.side !== 'top') edges.push([x0, z1, x1, z1])
  if (rect.side !== 'bottom') edges.push([x0, z0, x1, z0])
  if (rect.side !== 'left') edges.push([x1, z0, x1, z1])
  if (rect.side !== 'right') edges.push([x0, z0, x0, z1])
  // keep only edges that are away from the building: for a deck on 'bottom', the building is at z0
  const outer = edges.filter(([ax, az, bx, bz]) => {
    if (rect.side === 'bottom') return !(az === z0 && bz === z0)
    if (rect.side === 'top') return !(az === z1 && bz === z1)
    if (rect.side === 'right') return !(ax === x0 && bx === x0)
    return !(ax === x1 && bx === x1)
  })

  // pieces of each edge that are not interrupted by a staircase
  const pieces: { a: [number, number]; b: [number, number] }[] = []
  for (const [ax, az, bx, bz] of outer) {
    const horiz = az === bz
    const lo = horiz ? ax : az, hi = horiz ? bx : bz, fix = horiz ? az : ax
    const cuts: [number, number][] = []
    for (const s of stairs) {
      const sLo = horiz ? s.x0 : s.y0, sHi = horiz ? s.x1 : s.y1
      const nLo = horiz ? s.y0 : s.x0, nHi = horiz ? s.y1 : s.x1
      if (fix >= nLo - 0.4 && fix <= nHi + 0.4 && sHi > lo && sLo < hi) cuts.push([Math.max(lo, sLo - 0.05), Math.min(hi, sHi + 0.05)])
    }
    cuts.sort((p, q) => p[0] - q[0])
    let cur = lo
    for (const [c0, c1] of cuts) {
      if (c0 - cur > 0.2) pieces.push(horiz ? { a: [cur, fix], b: [c0, fix] } : { a: [fix, cur], b: [fix, c0] })
      cur = Math.max(cur, c1)
    }
    if (hi - cur > 0.2) pieces.push(horiz ? { a: [cur, fix], b: [hi, fix] } : { a: [fix, cur], b: [fix, hi] })
  }

  // balustrade pickets as one instanced mesh
  const picketGeo = useMemo(() => new THREE.BoxGeometry(0.025, 0.9, 0.025), [])
  const pickets = useMemo(() => {
    const out: [number, number][] = []
    for (const { a, b } of pieces) {
      const len = Math.hypot(b[0] - a[0], b[1] - a[1])
      const n = Math.max(1, Math.floor(len / 0.12))
      for (let i = 0; i <= n; i++) {
        const t = i / n
        out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(pieces)])
  const inst = useRef<THREE.InstancedMesh>(null)
  useEffect(() => {
    const m = inst.current
    if (!m) return
    const o = new THREE.Object3D()
    pickets.forEach(([x, z], i) => { o.position.set(x, 0.55, z); o.updateMatrix(); m.setMatrixAt(i, o.matrix) })
    m.instanceMatrix.needsUpdate = true
  }, [pickets])

  // verandah roof over the deck, falling gently away from the building
  const roofGeo = useMemo(() => {
    if (!roofed) return null
    const ov = 0.2
    const n: [number, number] = rect.side === 'bottom' ? [0, 1] : rect.side === 'top' ? [0, -1]
      : rect.side === 'right' ? [1, 0] : [-1, 0]
    const tg: [number, number] = [Math.abs(n[1]), Math.abs(n[0])]
    const mx = (x0 + x1) / 2, mz = (z0 + z1) / 2
    const half = (n[0] !== 0 ? d : w) / 2          // half of the depth ... recomputed below
    void half
    const depth = n[0] !== 0 ? w : d
    const along = n[0] !== 0 ? d : w
    const inner: [number, number] = [mx - n[0] * depth / 2, mz - n[1] * depth / 2]
    const outerC: [number, number] = [mx + n[0] * (depth / 2 + ov), mz + n[1] * (depth / 2 + ov)]
    const hi = eaveY - 0.08
    const lo = hi - (depth + ov) * 0.09
    const pt = (c: [number, number], s: number, y: number): P3 =>
      [c[0] + tg[0] * s * (along / 2 + ov), y, c[1] + tg[1] * s * (along / 2 + ov)]
    return {
      g: slabGeometry([pt(inner, -1, hi), pt(inner, 1, hi), pt(outerC, 1, lo), pt(outerC, -1, lo)], 0.07,
        roofFinish.tileM, !!roofFinish.swapUV),
      lo,
    }
  }, [roofed, rect.side, x0, x1, z0, z1, w, d, eaveY, roofFinish.tileM, roofFinish.swapUV])

  const posts: [number, number][] = []
  if (roofGeo) {
    for (const { a, b } of pieces) {
      const len = Math.hypot(b[0] - a[0], b[1] - a[1])
      const n = Math.max(1, Math.ceil(len / 2.8))
      for (let i = 0; i <= n; i++) {
        const t = i / n
        posts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
      }
    }
  }

  return (
    <group>
      <Box x0={x0} z0={z0} x1={x1} z1={z1} y0={-0.12} y1={0} color={TIMBER} />
      <mesh rotation-x={-Math.PI / 2} position={[(x0 + x1) / 2, 0.003, (z0 + z1) / 2]} receiveShadow>
        <planeGeometry args={[w, d]} />
        <meshStandardMaterial map={tex} roughness={0.85} />
      </mesh>

      {pieces.map(({ a, b }, i) => {
        const horiz = a[1] === b[1]
        const rail = (y0: number, y1: number) => horiz
          ? <Box key={`r${i}-${y0}`} x0={a[0]} x1={b[0]} z0={a[1] - 0.03} z1={a[1] + 0.03} y0={y0} y1={y1} color={STEEL} />
          : <Box key={`r${i}-${y0}`} x0={a[0] - 0.03} x1={a[0] + 0.03} z0={a[1]} z1={b[1]} y0={y0} y1={y1} color={STEEL} />
        return <group key={i}>{rail(1.0, 1.06)}{rail(0.08, 0.13)}</group>
      })}
      {pickets.length > 0 && (
        <instancedMesh key={pickets.length} ref={inst} args={[picketGeo, undefined, pickets.length]} castShadow>
          <meshStandardMaterial color={STEEL} metalness={0.5} roughness={0.4} />
        </instancedMesh>
      )}

      {roofGeo && (
        <>
          <mesh geometry={roofGeo.g} castShadow receiveShadow>
            <Surface finish={roofFinish} tint={roofColor} side={THREE.DoubleSide} />
          </mesh>
          {posts.map(([x, z], i) => (
            <Box key={`p${i}`} x0={x - 0.05} x1={x + 0.05} z0={z - 0.05} z1={z + 0.05}
                 y0={0} y1={roofGeo.lo} color={STEEL} />
          ))}
        </>
      )}
    </group>
  )
}

// ---------- Viewer ----------
interface ViewerProps {
  model: PlanModel
  wallH: number
  heights: Record<number, number>
  thicknesses: Record<number, number>
  deleted: number[]
  selectedIds: number[]
  onSelect: (id: number | null, additive?: boolean) => void
  dimMode: DimMode
  dimensionUnit?: DimensionUnit
  ground: GroundType
  groups: WallGroups
  roof: RoofConfig
  stumpM: number
  spec: OpeningSpec
  finish: FinishConfig
  extensions: Record<number, Ext>
  onExtend: (gid: number, e: Ext) => void
  snapOn: boolean
}

export default function Viewer({
  model, wallH, heights = {}, thicknesses = {}, deleted = [], selectedIds = [], onSelect,
  dimMode, dimensionUnit = 'ft-in', ground, groups, roof, stumpM, spec, finish, extensions = NO_EXT, onExtend, snapOn,
}: ViewerProps) {
  const b = model.bounds
  const cx = (b.x0 + b.x1) / 2
  const cz = (b.y0 + b.y1) / 2
  const size = Math.max(b.x1 - b.x0, b.y1 - b.y0, 4)

  const walls = useMemo(() => applyExtensions(model.walls, groups, extensions), [model.walls, groups, extensions])
  const visibleWalls = walls.filter(w => !deleted.includes(w.id))

  // the one wall being edited gets its two end handles
  const selGroups = Array.from(new Set(selectedIds.map(id => groups.groupOf[id] ?? id)))
  const editGid = selGroups.length === 1 ? selGroups[0] : null
  const axisInfo = editGid !== null ? groupAxis(model.walls, groups, editGid) : null

  const roofBox: Box2 = visibleWalls.length
    ? {
        x0: Math.min(...visibleWalls.map(w => w.x0)),
        z0: Math.min(...visibleWalls.map(w => w.y0)),
        x1: Math.max(...visibleWalls.map(w => w.x1)),
        z1: Math.max(...visibleWalls.map(w => w.y1)),
      }
    : { x0: b.x0, z0: b.y0, x1: b.x1, z1: b.y1 }

  const floorBox: Box2 = {
    x0: Math.min(b.x0, roofBox.x0), z0: Math.min(b.y0, roofBox.z0),
    x1: Math.max(b.x1, roofBox.x1), z1: Math.max(b.y1, roofBox.z1),
  }

  // ----- roof + deck from the PDF (only when the backend found them) -----
  const design = (model as PlanModel & { design?: DesignInfo }).design
  const useDesign = (roof.type as string) === 'design' && !!design?.roof
  const bx = roofBox.x1 - roofBox.x0
  const bz = roofBox.z1 - roofBox.z0
  const ridgeLen = design?.roof?.ridgeLengthM ?? null
  const ridgeAxis: 'x' | 'z' = ridgeLen
    ? (Math.abs(bx - ridgeLen) <= Math.abs(bz - ridgeLen) ? 'x' : 'z')
    : (bx >= bz ? 'x' : 'z')
  // if the plan was not scaled to real size, scale the PDF dimensions to the drawn building
  let k = 1
  if (ridgeLen) {
    k = (ridgeAxis === 'x' ? bx : bz) / ridgeLen
    if (!(k > 0.3 && k < 3) || Math.abs(k - 1) < 0.08) k = 1
  }
  let deckRect: DeckRect | null = null
  if (design?.deck && visibleWalls.length) {
    const side = design.deck.side
    const [da, db] = design.deck.dimsMm.map(v => (v / 1000) * k)
    const ext = side === 'top' || side === 'bottom' ? bx : bz
    let along = Math.abs(da - ext) <= Math.abs(db - ext) ? da : db
    const depth = along === da ? db : da
    if (Math.abs(along - ext) / ext > 0.35) along = ext
    const mid = side === 'top' || side === 'bottom' ? (roofBox.x0 + roofBox.x1) / 2 : (roofBox.z0 + roofBox.z1) / 2
    deckRect = side === 'bottom'
      ? { x0: mid - along / 2, x1: mid + along / 2, z0: roofBox.z1, z1: roofBox.z1 + depth, side, along, depth }
      : side === 'top'
      ? { x0: mid - along / 2, x1: mid + along / 2, z0: roofBox.z0 - depth, z1: roofBox.z0, side, along, depth }
      : side === 'right'
      ? { z0: mid - along / 2, z1: mid + along / 2, x0: roofBox.x1, x1: roofBox.x1 + depth, side, along, depth }
      : { z0: mid - along / 2, z1: mid + along / 2, x0: roofBox.x0 - depth, x1: roofBox.x0, side, along, depth }
  }

  const visibleOpenings = model.openings.filter(o => {
    const gid = groups.openingGroup[o.id]
    return gid === undefined || !deleted.includes(gid)
  })

  return (
    <Canvas
      shadows
      onPointerMissed={e => { if (!(e.ctrlKey || e.metaKey)) onSelect(null) }}
      key={`${cx.toFixed(2)}-${cz.toFixed(2)}-${model.walls.length}`}
      camera={{ position: [cx + size * 0.8, size * 0.9, cz + size * 1.1], fov: 50, far: 2000 }}
      style={{ background: '#bcd4e6' }}
    >
      <Sky
        sunPosition={[0.7, 0.4, 0.5]}
        turbidity={2}
        rayleigh={0.5}
        mieCoefficient={0.005}
        mieDirectionalG={0.8}
      />
      <fog attach="fog" args={['#bcd4e6', size * 3, size * 9]} />

      <Lights cx={cx} cz={cz} size={size} />

      <Environment resolution={256}>
        <Lightformer form="rect" intensity={3} position={[0, 6, -6]} scale={[14, 6, 1]} />
        <Lightformer form="rect" intensity={2} position={[-8, 3, 2]} rotation-y={Math.PI / 2} scale={[10, 4, 1]} />
        <Lightformer form="rect" intensity={2} position={[8, 3, 2]} rotation-y={-Math.PI / 2} scale={[10, 4, 1]} />
        <Lightformer form="ring" intensity={2} position={[0, 8, 0]} rotation-x={Math.PI / 2} scale={6} />
      </Environment>

      <Suspense fallback={null}>
        <Ground type={ground} cx={cx} cz={cz} size={size} />
      </Suspense>

      <FinishContext.Provider value={finish}>
        <group position={[0, stumpM, 0]}>
          {/* Floor slab */}
          <Box
            x0={floorBox.x0}
            z0={floorBox.z0}
            x1={floorBox.x1}
            z1={floorBox.z1}
            y0={-0.1}
            y1={0}
            color="#2b2d34"
          />

          {/* Walls: cut full-height wall meshes around openings; OpeningMesh renders lintels/sills. */}
          {visibleWalls.flatMap(w => {
            const t = thicknesses[w.id] ?? w.thickness
            const H = heights[w.id] ?? wallH
            const pick = (e: ThreeEvent<MouseEvent>) => {
              e.stopPropagation()
              onSelect(w.id, e.nativeEvent.ctrlKey || e.nativeEvent.metaKey)
            }

            // Keep diagonal walls intact; axis-aligned runs are cut at their openings.
            if (w.diagonal) {
              const ax = w.ax as number, ay = w.ay as number, bx = w.bx as number, by = w.by as number
              const len = Math.hypot(bx - ax, by - ay)
              return [
                <group key={w.id} position={[(ax + bx) / 2, 0, (ay + by) / 2]}
                       rotation-y={-Math.atan2(by - ay, bx - ax)}>
                  <Box x0={-len / 2} x1={len / 2} z0={-t / 2} z1={t / 2}
                       y0={0} y1={H} color={WALL} glossy
                       selected={selectedIds.includes(w.id)} onClick={pick} />
                </group>
              ]
            }

            const mx = (w.x0 + w.x1) / 2
            const mz = (w.y0 + w.y1) / 2
            const x0 = w.horizontal ? w.x0 : mx - t / 2
            const x1 = w.horizontal ? w.x1 : mx + t / 2
            const z0 = w.horizontal ? mz - t / 2 : w.y0
            const z1 = w.horizontal ? mz + t / 2 : w.y1
            const segments = wallLengthSegments(w, visibleOpenings)

            return segments.map(([a, b], i) => (
              <Box
                key={`${w.id}-segment-${i}`}
                x0={w.horizontal ? a : x0}
                z0={w.horizontal ? z0 : a}
                x1={w.horizontal ? b : x1}
                z1={w.horizontal ? z1 : b}
                y0={0} y1={H}
                color={WALL}
                glossy
                shrink={w.horizontal ? 0 : 0.004}
                selected={selectedIds.includes(w.id)}
                onClick={pick}
              />
            ))
          })}

          {editGid !== null && axisInfo && (
            <ExtendHandles
              key={editGid}
              gid={editGid}
              axis={axisInfo}
              ext={extensions[editGid] ?? { start: 0, end: 0 }}
              wallH={heights[axisInfo.startId] ?? wallH}
              stumpM={stumpM}
              snapOn={snapOn}
              walls={visibleWalls}
              skip={groups.members[editGid] ?? []}
              onChange={onExtend}
            />
          )}

          {/* Railings */}
          {model.railings?.map(r => (
            <RailingMesh key={r.id} r={r} height={1.0} />
          ))}

          {/* NEW: Stairs */}
          {model.stairs?.map(stair => (
            <StairsMesh key={stair.id} stair={stair} height={1.2} />
          ))}

          {/* Doors and windows */}
          {visibleOpenings.map(o => {
            const gid = groups.openingGroup[o.id]
            const has = gid !== undefined
            const H = has ? (heights[gid] ?? wallH) : wallH
            const t = has ? (thicknesses[gid] ?? o.thickness) : o.thickness

            return (
              <OpeningMesh
                key={o.id}
                o={{ ...o, thickness: t }}
                H={H}
                spec={spec}
                selected={has && selectedIds.includes(gid)}
                onPick={
                  has
                    ? e => {
                        e.stopPropagation()
                        onSelect(gid, e.nativeEvent.ctrlKey || e.nativeEvent.metaKey)
                      }
                    : undefined
                }
              />
            )
          })}

          {/* Dimensions */}
          <Dimensions
            model={{
              ...model,
              walls: visibleWalls,
              openings: visibleOpenings,
            }}
            H={wallH}
            mode={dimMode}
            scale={size * 1.6}
            unit={dimensionUnit}
          />

          {/* Deck from the PDF */}
          {useDesign && deckRect && (
            <Deck rect={deckRect} stairs={model.stairs ?? []} eaveY={wallH}
                  roofed={!!design?.deck?.roofed && !!design?.roof?.verandah} />
          )}

          {/* Roof */}
          {useDesign && design?.roof?.type === 'gable' ? (
            roof.visible && (
              <DesignRoof box={roofBox} eaveY={wallH} riseM={design.roof.riseM * k}
                          axis={ridgeAxis} overhang={roof.overhangM} />
            )
          ) : (
            <Roof
              cfg={useDesign && design?.roof
                ? { ...roof, type: design.roof.type as RoofType, ridgeM: wallH + design.roof.riseM * k }
                : roof}
              box={roofBox}
              eaveY={wallH}
            />
          )}
        </group>
      </FinishContext.Provider>

      {/* Supporting stumps */}
      <Stumps box={roofBox} height={stumpM} />
      {useDesign && deckRect && <Stumps box={deckRect} height={stumpM} />}

      {/* Camera controls */}
      <OrbitControls
        makeDefault
        enableDamping
        target={[cx, stumpM + wallH / 2, cz]}
      />
    </Canvas>
  )
}