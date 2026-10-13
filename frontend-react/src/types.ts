export interface Wall {
  id: number
  x0: number; y0: number; x1: number; y1: number
  horizontal: boolean
  length: number
  thickness: number
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
}

export type DimMode = 'off' | 'overall' | 'all'