// Site placement: where on Earth the plan stands, how it is turned, and the plot boundary.
// Pure helpers (no Leaflet), shared by the map picker and the 3D viewer.
import * as THREE from 'three'

export interface SiteConfig {
  lat: number                      // the middle of the plan
  lng: number
  rotationDeg: number              // plan turned clockwise from "top of the plan = north"
  boundary: [number, number][]     // plot corners as [lat, lng]; may be empty
  layer: 'satellite' | 'street'
  zoom: number                     // tile detail used for the 3D ground (18 normal, 19 sharp)
}

export const TILE_URL = {
  satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  street: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
}
export const ATTRIBUTION = {
  satellite: 'Imagery © Esri, Maxar, Earthstar Geographics',
  street: '© OpenStreetMap contributors',
}
export const SITE_EXTENT = 300         // metres of map laid under the model
const M_PER_DEG = 111320

export const mPerPx = (lat: number, z: number) => 156543.03392 * Math.cos(lat * Math.PI / 180) / 2 ** z

/** Web-Mercator "world pixel" of a point at zoom z. */
export function worldPx(lat: number, lng: number, z: number): [number, number] {
  const s = 256 * 2 ** z
  const sin = Math.sin(lat * Math.PI / 180)
  return [((lng + 180) / 360) * s, (0.5 - Math.log((1 + sin) / (1 - sin)) / (4 * Math.PI)) * s]
}

/** metres east and south of (lat0, lng0) */
export function toLocal(lat0: number, lng0: number, lat: number, lng: number): [number, number] {
  return [(lng - lng0) * M_PER_DEG * Math.cos(lat0 * Math.PI / 180), (lat0 - lat) * M_PER_DEG]
}
export function fromLocal(lat0: number, lng0: number, east: number, south: number): [number, number] {
  return [lat0 - south / M_PER_DEG, lng0 + east / (M_PER_DEG * Math.cos(lat0 * Math.PI / 180))]
}

/** The four corners of the plan (width x depth metres) on the map, as [lat, lng]. */
export function footprint(lat: number, lng: number, w: number, d: number, rotDeg: number): [number, number][] {
  const r = rotDeg * Math.PI / 180, c = Math.cos(r), s = Math.sin(r)
  return ([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]] as [number, number][])
    .map(([x, y]) => fromLocal(lat, lng, x * c - y * s, x * s + y * c))
}

export function insidePolygon(p: [number, number], poly: [number, number][]) {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [yi, xi] = poly[i], [yj, xj] = poly[j]
    if ((yi > p[0]) !== (yj > p[0]) && p[1] < ((xj - xi) * (p[0] - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

/** Plot area in square metres (shoelace on local metres). */
export function polygonAreaM2(poly: [number, number][]) {
  if (poly.length < 3) return 0
  const pts = poly.map(([la, ln]) => toLocal(poly[0][0], poly[0][1], la, ln))
  let a = 0
  for (let i = 0; i < pts.length; i++) { const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length]; a += x1 * y2 - x2 * y1 }
  return Math.abs(a) / 2
}

/** Stitch map tiles into one canvas texture, centred on the site, north at the top. */
export async function buildSiteTexture(site: SiteConfig, extent = SITE_EXTENT) {
  const z = site.zoom
  const mpp = mPerPx(site.lat, z)
  const size = Math.min(4096, Math.ceil(extent / mpp))
  const usedExtent = size * mpp
  const [cx, cy] = worldPx(site.lat, site.lng, z)
  const left = cx - size / 2, top = cy - size / 2
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const g = canvas.getContext('2d')!
  g.fillStyle = '#6f7d57'
  g.fillRect(0, 0, size, size)
  const n = 2 ** z
  const jobs: Promise<void>[] = []
  for (let ty = Math.floor(top / 256); ty <= Math.floor((top + size) / 256); ty++) {
    for (let tx = Math.floor(left / 256); tx <= Math.floor((left + size) / 256); tx++) {
      if (ty < 0 || ty >= n) continue
      const url = TILE_URL[site.layer].replace('{z}', String(z)).replace('{x}', String(((tx % n) + n) % n)).replace('{y}', String(ty))
      jobs.push(new Promise<void>(res => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => { g.drawImage(img, tx * 256 - left, ty * 256 - top, 256, 256); res() }
        img.onerror = () => res()
        img.src = url
      }))
    }
  }
  await Promise.all(jobs)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace
  tex.anisotropy = 8
  tex.needsUpdate = true
  return { texture: tex, extent: usedExtent }
}