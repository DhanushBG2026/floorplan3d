
import { useState } from 'react'
import ReactCrop, { type Crop } from 'react-image-crop'
import 'react-image-crop/dist/ReactCrop.css'
import {
  DEFAULT_ADJUSTMENTS,
  type ImageAdjustments,
} from './api'

// x0, y0, x1, y1 as fractions of the page (0..1)
export type CropBox = [number, number, number, number]

interface Props {
  image: string
  initial: CropBox | null
  adjustments: ImageAdjustments
  onAdjustmentsChange: (value: ImageAdjustments) => void
  onConfirm: (box: CropBox | null) => void
  onCancel: () => void
}

const sliders = [
  { key: 'brightness', label: 'Brightness', min: -100, max: 100 },
  { key: 'contrast', label: 'Contrast', min: -100, max: 100 },
  { key: 'highlights', label: 'Highlights', min: -100, max: 100 },
  { key: 'shadows', label: 'Shadows', min: -100, max: 100 },
  { key: 'whites', label: 'Whites', min: -100, max: 100 },
  { key: 'blacks', label: 'Blacks', min: -100, max: 100 },
  { key: 'sharpness', label: 'Sharpness', min: 0, max: 100 },
] as const

export default function CropDialog({
  image,
  initial,
  adjustments,
  onAdjustmentsChange,
  onConfirm,
  onCancel,
}: Props) {
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
    onConfirm([
      crop.x / 100,
      crop.y / 100,
      (crop.x + crop.width) / 100,
      (crop.y + crop.height) / 100,
    ])
  }

  const update = (key: keyof ImageAdjustments, value: number | boolean) => {
    onAdjustmentsChange({ ...adjustments, [key]: value })
  }

  // Fast visual preview. The backend applies the actual tonal adjustments
  // to the pixels used for detection.
  const previewFilter = [
    `brightness(${1 + adjustments.brightness / 200})`,
    `contrast(${1 + adjustments.contrast / 100})`,
    adjustments.invert ? 'invert(1)' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>Prepare floor plan</h2>
        <p>
          Select the plan area, then adjust its tones to make walls
          and openings easier to detect.
        </p>

        <div className="crop-area">
          <ReactCrop
            crop={crop}
            onChange={(_, pct) => setCrop(pct)}
            minWidth={30}
            minHeight={30}
          >
            <img
              src={image}
              alt="Uploaded floor plan"
              style={{ filter: previewFilter }}
            />
          </ReactCrop>
        </div>

        <section className="image-adjustments">
          <div className="adjustments-heading">
            <h3>Image adjustments</h3>
            <button
              type="button"
              className="ghost"
              onClick={() => onAdjustmentsChange({ ...DEFAULT_ADJUSTMENTS })}
            >
              Reset all
            </button>
          </div>

          {sliders.map(({ key, label, min, max }) => (
            <label className="adjustment-control" key={key}>
              <span>
                {label}
                <output>
                  {adjustments[key] > 0 ? '+' : ''}
                  {adjustments[key]}
                </output>
              </span>
              <input
                type="range"
                min={min}
                max={max}
                step={1}
                value={adjustments[key]}
                onChange={e => update(key, Number(e.target.value))}
              />
            </label>
          ))}

          <label className="adjustment-toggle">
            <input
              type="checkbox"
              checked={adjustments.invert}
              onChange={e => update('invert', e.target.checked)}
            />
            Invert image
          </label>
        </section>

        <div className="modal-actions">
          <button className="ghost" onClick={onCancel}>Cancel</button>
          <button
            className="ghost"
            onClick={() => onConfirm(null)}
          >
            Use full page
          </button>
          <button onClick={confirm} disabled={!valid}>
            Use selected area
          </button>
        </div>
      </div>
    </div>
  )
}
