export interface Finish {
  label: string
  color: string      // colour of a plain finish, or the tint laid over a texture
  map?: string       // colour texture in /public/textures
  normal?: string    // OpenGL normal map
  tileM: number      // metres covered by one repeat of the texture
  swapUV?: boolean   // turn the pattern 90 degrees
  roughness: number
  metalness: number
}

export type WallFinishKey = 'plain' | 'plaster' | 'cladding' | 'brick'
export type RoofFinishKey = 'metal' | 'tiles' | 'plain'

export const DEFAULT_ROOF_COLOR = '#6b4430' // dark brown

export const WALL_FINISHES: Record<WallFinishKey, Finish> = {
  plain: { label: 'Plain', color: '#e8e4dc', tileM: 2, roughness: 0.25, metalness: 0.05 },
  plaster: { label: 'Plaster', color: '#ffffff', map: '/textures/plaster_color.jpg', normal: '/textures/plaster_normal.jpg', tileM: 2, roughness: 0.9, metalness: 0 },
  cladding: { label: 'Cladding', color: '#ffffff', map: '/textures/cladding_color.jpg', normal: '/textures/cladding_normal.jpg', tileM: 1.5, roughness: 0.8, metalness: 0 },
  brick: { label: 'Brick', color: '#ffffff', map: '/textures/brick_color.jpg', normal: '/textures/brick_normal.jpg', tileM: 1.2, roughness: 0.9, metalness: 0 },
}

export const ROOF_FINISHES: Record<RoofFinishKey, Finish> = {
  metal: { label: 'Corrugated metal', color: '#ffffff', map: '/textures/roof_metal_color.jpg', normal: '/textures/roof_metal_normal.jpg', tileM: 1.5, roughness: 0.5, metalness: 0.4 },
  tiles: { label: 'Tiles', color: '#ffffff', map: '/textures/roof_tiles_color.jpg', normal: '/textures/roof_tiles_normal.jpg', tileM: 1.2, roughness: 0.75, metalness: 0 },
  plain: { label: 'Plain colour', color: DEFAULT_ROOF_COLOR, tileM: 1, roughness: 0.55, metalness: 0.2 },
}