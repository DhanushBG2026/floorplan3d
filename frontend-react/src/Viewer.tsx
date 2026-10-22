import { Component, Suspense, createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from 'react'
import * as THREE from 'three'
import { Canvas, useThree, type ThreeEvent } from '@react-three/fiber'
import {
  OrbitControls, Html, Line, Edges, useTexture, Environment, Lightformer, Sky,
} from '@react-three/drei'
import type { PlanModel, Opening, DimMode } from './types'
import type { WallGroups } from './groups'
import { GROUNDS, type GroundType } from './ground'
import { fmtFtIn } from './units'
import { roofTriangles, planarUVs, type RoofType } from './roof'
import { WALL_FINISHES, ROOF_FINISHES, DEFAULT_ROOF_COLOR, type Finish } from './materials'


export interface RoofConfig { type: RoofType; visible: boolean; ridgeM: number; overhangM: number }
export interface OpeningSpec { sillM: number; headM: number; doorM: number }
interface Box2 { x0: number; z0: number; x1: number; z1: number }


const DOOR_WOOD = '#8a5a32'
const DOOR_FRAME = '#5c3b1e'

const DOOR_H = 2.1
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
  const size: [number, number][] = [[d, h], [d, h], [w, d], [w, d], [w, h], [w, h]] // +x -x +y -y +z -z
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
  const dh = Math.min(doorH, h)   // never taller than the wall
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
      {part(o.start + f, o.end - f, c - leafT / 2, c + leafT / 2, 0.01, dh - 0.01, DOOR_WOOD, true)}
      {knob(1)}
      {knob(-1)}
    </>
  )
}

interface OpeningMeshProps {
  o: Opening
  H: number
  selected?: boolean
  onPick?: (e: ThreeEvent<MouseEvent>) => void
}

interface OpeningMeshProps {
  o: Opening
  H: number
  spec: OpeningSpec
  selected?: boolean
  onPick?: (e: ThreeEvent<MouseEvent>) => void
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
  const top = Math.min(o.type === 'window' ? spec.headM : spec.doorM, H)
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
      {o.type === 'door' && <DoorParts o={o} h={H} doorH={spec.doorM} />}
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

function Dimensions({
  model, H, mode, scale,
}: { model: PlanModel; H: number; mode: DimMode; scale: number }) {
  if (mode === 'off') return null
  const b = model.bounds
  const cx = (b.x0 + b.x1) / 2
  const cz = (b.y0 + b.y1) / 2
  const gap = 1.2
  const fmt = (v: number) => fmtFtIn(v)
  

  return (
    <>
      {/* overall width (bottom) and depth (left), on the ground outside the plan */}
      <Dim kind="overall" scale={scale} a={[b.x0, 0.03, b.y1]} b={[b.x1, 0.03, b.y1]}
           offset={[0, 0, gap]} text={fmt(b.x1 - b.x0)} />
      <Dim kind="overall" scale={scale} a={[b.x0, 0.03, b.y0]} b={[b.x0, 0.03, b.y1]}
           offset={[-gap, 0, 0]} text={fmt(b.y1 - b.y0)} />

      {/* every wall length, only in "all" mode, skipping short stubs */}
      {mode === 'all' &&
        model.walls.filter(w => w.length >= 1.5).map(w => {
          const hz = w.horizontal
          const mid = hz ? (w.y0 + w.y1) / 2 : (w.x0 + w.x1) / 2
          const outward = hz ? (mid < cz ? -1 : 1) : (mid < cx ? -1 : 1)
          const off = (w.thickness / 2 + 0.35) * outward
          const a: V3 = hz ? [w.x0, H + 0.02, mid] : [mid, H + 0.02, w.y0]
          const c: V3 = hz ? [w.x1, H + 0.02, mid] : [mid, H + 0.02, w.y1]
          const offset: V3 = hz ? [0, 0, off] : [off, 0, 0]
          return <Dim key={`w${w.id}`} kind="wall" scale={scale} a={a} b={c} offset={offset} text={fmt(w.length)} />
        })}

      {/* doors, windows, openings: label just above the wall top */}
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
  const h = height - 0.1 // the floor slab is 0.1 m thick and sits on top of the posts
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
  ground: GroundType
  groups: WallGroups
  roof: RoofConfig
  stumpM: number
  spec: OpeningSpec
  finish: FinishConfig
}

 export default function Viewer({
  model, wallH, heights = {}, thicknesses = {}, deleted = [], selectedIds = [], onSelect, dimMode, ground, groups, roof, stumpM, spec, finish,
  }: ViewerProps) {
  const b = model.bounds
  const cx = (b.x0 + b.x1) / 2
  const cz = (b.y0 + b.y1) / 2
  const size = Math.max(b.x1 - b.x0, b.y1 - b.y0, 4)  
  const visibleWalls = model.walls.filter(w => !deleted.includes(w.id))
  const roofBox: Box2 = visibleWalls.length
    ? {
        x0: Math.min(...visibleWalls.map(w => w.x0)),
        z0: Math.min(...visibleWalls.map(w => w.y0)),
        x1: Math.max(...visibleWalls.map(w => w.x1)),
        z1: Math.max(...visibleWalls.map(w => w.y1)),
      }
    : { x0: b.x0, z0: b.y0, x1: b.x1, z1: b.y1 }
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
            x0={b.x0}
            z0={b.y0}
            x1={b.x1}
            z1={b.y1}
            y0={-0.1}
            y1={0}
            color="#2b2d34"
          />

          {/* Walls */}
          {visibleWalls.map(w => {
            const t = thicknesses[w.id] ?? w.thickness
            const mx = (w.x0 + w.x1) / 2
            const mz = (w.y0 + w.y1) / 2

            const x0 = w.horizontal ? w.x0 : mx - t / 2
            const x1 = w.horizontal ? w.x1 : mx + t / 2
            const z0 = w.horizontal ? mz - t / 2 : w.y0
            const z1 = w.horizontal ? mz + t / 2 : w.y1

            return (
              <Box
                key={w.id}
                x0={x0}
                z0={z0}
                x1={x1}
                z1={z1}
                y0={0}
                y1={heights[w.id] ?? wallH}
                color={WALL}
                edges
                glossy
                shrink={w.horizontal ? 0 : 0.004}
                selected={selectedIds.includes(w.id)}
                onClick={e => {
                  e.stopPropagation()
                  onSelect(
                    w.id,
                    e.nativeEvent.ctrlKey || e.nativeEvent.metaKey
                  )
                }}
              />
            )
          })}

          {/* Doors and windows */}
          {visibleOpenings.map(o => {
            const gid = groups.openingGroup[o.id]
            const has = gid !== undefined
            const H = has ? (heights[gid] ?? wallH) : wallH
            const t = has
              ? (thicknesses[gid] ?? o.thickness)
              : o.thickness

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
                        onSelect(
                          gid,
                          e.nativeEvent.ctrlKey ||
                            e.nativeEvent.metaKey
                        )
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
          />

          {/* Roof */}
          <Roof cfg={roof} box={roofBox} eaveY={wallH} />
        </group>
      </FinishContext.Provider>

      {/* Supporting stumps */}
      <Stumps box={roofBox} height={stumpM} />

      {/* Camera controls */}
      <OrbitControls
        makeDefault
        enableDamping
        target={[cx, stumpM + wallH / 2, cz]}
      />

    </Canvas>
  )
}