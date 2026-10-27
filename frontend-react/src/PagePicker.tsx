// Choose which page of a multipage PDF holds the floor plan. Opens before the crop dialog.
export interface PageThumb { index: number; thumb: string; kind: string; label: string }

interface Props {
  pages: PageThumb[]
  current: number                    // page shown now (the one found automatically at first)
  suggested?: number | null          // page the backend thinks is the floor plan
  onChoose: (index: number) => void
  onCancel: () => void
}

export default function PagePicker({ pages, current, suggested, onChoose, onCancel }: Props) {
  return (
    <div className="modal-backdrop">
      <div className="modal plan-editor-modal">
        <header className="plan-editor-header">
          <div>
            <h2>Choose the floor plan page</h2>
            <p>This PDF has {pages.length} pages. Pick the page that holds the floor plan; you can crop it next.</p>
          </div>
        </header>

        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 14,
          padding: 16, overflow: 'auto', flex: 1,
        }}>
          {pages.map(p => (
            <button
              key={p.index}
              type="button"
              className="ghost"
              onClick={() => onChoose(p.index)}
              style={{
                display: 'flex', flexDirection: 'column', gap: 8, padding: 8, textAlign: 'left', cursor: 'pointer',
                border: p.index === current ? '2px solid #2b6cb0' : '1px solid #cfcfcf', borderRadius: 8, background: '#fff',
              }}
            >
              <img src={p.thumb} alt={`Page ${p.index + 1}`} style={{ width: '100%', display: 'block', border: '1px solid #e2e2e2' }} />
              <span style={{ display: 'flex', justifyContent: 'space-between', gap: 6, fontSize: 13 }}>
                <span>Page {p.index + 1}: {p.label}</span>
                {p.index === suggested && <strong style={{ color: '#2b6cb0' }}>Suggested</strong>}
              </span>
            </button>
          ))}
        </div>

        <footer className="plan-editor-footer">
          <button className="ghost" onClick={onCancel}>Cancel</button>
          <button onClick={() => onChoose(current)}>Use page {current + 1}</button>
        </footer>
      </div>
    </div>
  )
}