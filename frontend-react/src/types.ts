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

export type DimMode = 'off' | 'overall' | 'all'