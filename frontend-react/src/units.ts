export const FT = 0.3048 // metres per foot

export const toFt = (m: number) => m / FT
export const toM = (ft: number) => ft * FT

// 3.81 m -> 12' 6"
export function fmtFtIn(m: number): string {
  const totalIn = Math.round(m / 0.0254)
  const ft = Math.floor(totalIn / 12)
  const inch = totalIn % 12
  return `${ft}' ${inch}"`
}