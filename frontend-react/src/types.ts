export interface Wall {
  id: number
  x0: number
  y0: number
  x1: number
  y1: number
  horizontal: boolean
  length: number
  thickness: number
  // diagonal walls (ML detector only)
  diagonal?: boolean
  ax?: number
  ay?: number
  bx?: number
  by?: number
  arc?: { cx: number; cy: number; r: number; a0: number; a1: number }
  bay?: boolean
}

export interface Opening {
  id: number
  type: 'door' | 'window' | 'opening'
  horizontal: boolean
  start: number
  end: number
  centre: number
  thickness: number
  width: number
  ink_density: number
  
}

export interface PlanModel {
  units: string
  px_per_m: number
  bounds: { x0: number; y0: number; x1: number; y1: number }
  walls: Wall[]
  openings: Opening[]
  overlay: string
  railings?: Railing[]
  stairs?: Stair[]
  design?: import('./Viewer').DesignInfo
  plan?: { src: string; wM: number; hM: number }
  floors?: FloorInfo[]
  decks?: { id: number; x0: number; y0: number; x1: number; y1: number; side: 'top' | 'bottom' | 'left' | 'right'; roofed?: boolean }[]
}

export interface Railing {
  id: number
  x0: number
  y0: number
  x1: number
  y1: number
  horizontal: boolean
  length: number
  thickness: number
}

export interface Stair {
  id: number
  x0: number
  y0: number
  x1: number
  y1: number
  horizontal: boolean
  length: number
  width: number
  steps: number
}

export interface FloorInfo {
  id: number; label: string
  bbox: { x0: number; y0: number; x1: number; y1: number }
  wallIds: number[]; openingIds: number[]; stairIds: number[]; railingIds: number[]
  similar?: boolean
  shiftTo: Record<string, [number, number]>
}

export type DimMode = 'off' | 'overall' | 'all'