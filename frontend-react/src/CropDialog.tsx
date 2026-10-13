import { useState } from 'react'
import ReactCrop, { type Crop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'

// x0, y0, x1, y1 as fractions of the page (0..1)
export type CropBox = [number, number, number, number]

interface Props {
  image: string
  initial: CropBox | null
  onConfirm: (box: CropBox | null) => void
  onCancel: () => void
}

export default function CropDialog({ image, initial, onConfirm, onCancel }: Props) {
  const [crop, setCrop] = useState<Crop | undefined>(
    initial
      ? {
          unit: '%',
          x: initial[0] * 100,
          y: initial[1] * 100,
          width: (initial[2] - initial[0]) * 100,
          height: (initial[3] - initial[1]) * 100,
        }
      : undefined,
  )
  const valid = !!crop && crop.width >= 5 && crop.height >= 5

  const confirm = () => {
    if (!crop || !valid) return
    onConfirm([crop.x / 100, crop.y / 100, (crop.x + crop.width) / 100, (crop.y + crop.height) / 100])
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>Select the floor plan</h2>
        <p>Drag a box around the floor plan only. Leave out the title block, legends, tables and notes.</p>
        <div className="crop-area">
          <ReactCrop crop={crop} onChange={(_, pct) => setCrop(pct)} minWidth={30} minHeight={30}>
            <img src={image} alt="Uploaded plan" />
          </ReactCrop>
        </div>
        <div className="modal-actions">
          <button className="ghost" onClick={onCancel}>Cancel</button>
          <button className="ghost" onClick={() => onConfirm(null)}>Use full page</button>
          <button onClick={confirm} disabled={!valid}>Use selected area</button>
        </div>
      </div>
    </div>
  )
}