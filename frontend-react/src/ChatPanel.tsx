import { useEffect, useRef, useState } from 'react'
import { chatWithDocument, type ChatMessage } from './api'

const SUGGESTIONS = [
  'What is the plot size?',
  'How big is the kitchen?',
  'What is the total area of the two bedrooms?',
]

interface Props {
  documentText: string
  docKey: string
}

export default function ChatPanel({ documentText, docKey }: Props) {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)
  const hasText = documentText.trim().length > 0

  // new document -> new conversation
  useEffect(() => { setMessages([]); setInput('') }, [docKey])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy, open])

  const send = async (text?: string) => {
    const q = (text ?? input).trim()
    if (!q || busy || !hasText) return
    const next: ChatMessage[] = [...messages, { role: 'user', content: q }]
    setMessages(next)
    setInput('')
    setBusy(true)
    try {
      const reply = await chatWithDocument(documentText, next)
      setMessages([...next, { role: 'assistant', content: reply }])
    } catch (e) {
      setMessages([...next, { role: 'assistant', content: (e as Error).message, error: true }])
    }
    setBusy(false)
  }

  if (!open) {
    return <button className="chat-fab" onClick={() => setOpen(true)}>💬 Ask about this plan</button>
  }

  return (
    <div className="chat">
      <div className="chat-head">
        <strong>Ask about this plan</strong>
        <button className="chat-close" onClick={() => setOpen(false)} aria-label="Close">✕</button>
      </div>

      <div className="chat-body">
        {!hasText ? (
          <p className="chat-note">
            No readable text was found in this file, so I can't answer questions about it.
            Chat works with PDFs that have selectable text.
          </p>
        ) : messages.length === 0 ? (
          <div className="chat-suggest">
            <p className="chat-note">Answers come only from the text in your uploaded document.</p>
            {SUGGESTIONS.map(s => (
              <button key={s} onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}${m.error ? ' error' : ''}`}>{m.content}</div>
          ))
        )}
        {busy && <div className="msg assistant">Thinking…</div>}
        <div ref={endRef} />
      </div>

      {hasText && (
        <details className="chat-source">
          <summary>Text the assistant can see</summary>
          <pre>{documentText}</pre>
        </details>
      )}

      <div className="chat-input">
        <input
          value={input}
          placeholder={hasText ? 'Ask a question…' : 'No text to ask about'}
          disabled={!hasText}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') send() }}
        />
        <button onClick={() => send()} disabled={busy || !hasText || !input.trim()}>Send</button>
      </div>
    </div>
  )
}