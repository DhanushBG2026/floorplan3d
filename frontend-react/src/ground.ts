export type GroundType = 'grass' | 'sand'

export const GROUNDS: Record<GroundType, { label: string; texture: string; tileM: number }> = {
  grass: { label: 'Grass', texture: '/textures/grass.jpg', tileM: 4 },
  sand: { label: 'Sand', texture: '/textures/sand.jpg', tileM: 3 },
}