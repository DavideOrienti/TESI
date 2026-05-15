import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'

const WELCOME = "Ciao! Sono CineBot 🎬 Dimmi che tipo di film ti va di vedere stasera, o chiedimi qualsiasi cosa sul cinema!"
const FALLBACK_POSTER = 'https://via.placeholder.com/60x90/1f2937/6b7280?text=?'

function MovieChip({ movie }) {
  const navigate = useNavigate()
  return (
    <div
      className="flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
      style={{ width: 64 }}
      onClick={() => navigate(`/movies/${movie.movie_id}`)}
      title={movie.reason || movie.title}
    >
      <img
        src={movie.poster_url || FALLBACK_POSTER}
        alt={movie.title}
        style={{ width: 60, height: 90, objectFit: 'cover', borderRadius: 8 }}
        onError={e => { e.target.src = FALLBACK_POSTER }}
      />
      <p
        className="line-clamp-2"
        style={{ fontSize: '0.6rem', color: '#cbd5e1', marginTop: 3, lineHeight: 1.2, textAlign: 'center' }}
      >
        {movie.title}
      </p>
    </div>
  )
}

function Message({ msg }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: isUser ? 'flex-end' : 'flex-start', marginBottom: 10 }}>
      <div
        className={isUser ? 'ml-8' : 'mr-8'}
        style={{
          maxWidth: '85%',
          padding: '8px 12px',
          borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
          background: isUser ? 'rgba(99,102,241,0.22)' : 'rgba(255,255,255,0.06)',
          border: isUser ? '1px solid rgba(99,102,241,0.3)' : '1px solid rgba(255,255,255,0.10)',
          color: isUser ? '#e2e8f0' : '#cbd5e1',
          fontSize: '0.78rem',
          lineHeight: 1.5,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {msg.content}
      </div>
      {msg.movies?.length > 0 && (
        <div style={{ display: 'flex', gap: 8, marginTop: 8, overflowX: 'auto', maxWidth: '100%', paddingBottom: 4 }}>
          {msg.movies.map(m => <MovieChip key={m.movie_id} movie={m} />)}
        </div>
      )}
    </div>
  )
}

export default function ChatBubble() {
  const [open, setOpen] = useState(false)
  const [messages, setMessages] = useState([
    { role: 'assistant', content: WELCOME, movies: [] }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [unread, setUnread] = useState(1)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (open) {
      setUnread(0)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    }
  }, [open])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', content: text, movies: [] }
    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    setInput('')
    setLoading(true)

    const history = nextMessages
      .slice(-11, -1)
      .map(m => ({ role: m.role, content: m.content }))

    try {
      const res = await api.post('/chat/message', { message: text, history })
      const botMsg = {
        role: 'assistant',
        content: res.data.response,
        movies: res.data.suggested_movies ?? []
      }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      const status = err.response?.status
      let errText = 'Ops, qualcosa è andato storto. Riprova!'
      if (status === 503) errText = 'Il servizio LLM non è disponibile al momento.'
      setMessages(prev => [...prev, { role: 'assistant', content: errText, movies: [] }])
    } finally {
      setLoading(false)
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <>
      {open && (
        <div
          style={{
            position: 'fixed',
            bottom: 88,
            right: 24,
            width: 360,
            height: 500,
            borderRadius: 24,
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            background: 'rgba(15,15,26,0.95)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            border: '1px solid rgba(255,255,255,0.12)',
            boxShadow: '0 25px 50px rgba(0,0,0,0.6)',
            animation: 'slideUp 0.2s ease-out',
          }}
        >
          {/* Header */}
          <div style={{
            padding: '12px 16px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            background: 'linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2))',
            borderBottom: '1px solid rgba(255,255,255,0.10)',
          }}>
            <span style={{ color: '#a5b4fc', fontWeight: 600, fontSize: '0.875rem' }}>🎬 CineBot</span>
            <button
              onClick={() => setOpen(false)}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '1.2rem', lineHeight: 1, padding: '2px 6px' }}
              className="hover:text-slate-200 transition-colors"
            >
              ×
            </button>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column' }}>
            {messages.map((msg, i) => <Message key={i} msg={msg} />)}
            {loading && (
              <div style={{ display: 'flex', alignItems: 'flex-start', marginBottom: 10 }}>
                <div style={{
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.10)',
                  borderRadius: '16px 16px 16px 4px',
                  padding: '8px 14px',
                }}>
                  <span style={{ color: '#64748b', fontSize: '0.78rem' }} className="animate-pulse">
                    CineBot sta scrivendo…
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div style={{
            padding: '12px',
            borderTop: '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            gap: 8,
            alignItems: 'flex-end',
          }}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Scrivi un messaggio… (Enter per inviare)"
              rows={1}
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.06)',
                color: '#e2e8f0',
                border: '1px solid rgba(255,255,255,0.10)',
                borderRadius: 12,
                padding: '8px 12px',
                fontSize: '0.78rem',
                resize: 'none',
                outline: 'none',
                maxHeight: 80,
                overflowY: 'auto',
                lineHeight: 1.4,
              }}
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              style={{
                background: loading || !input.trim()
                  ? 'rgba(255,255,255,0.06)'
                  : 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                color: loading || !input.trim() ? '#475569' : '#fff',
                border: 'none',
                borderRadius: 12,
                padding: '8px 14px',
                cursor: loading || !input.trim() ? 'default' : 'pointer',
                fontSize: '0.78rem',
                fontWeight: 600,
                transition: 'all 0.15s',
                whiteSpace: 'nowrap',
              }}
            >
              Invia
            </button>
          </div>
        </div>
      )}

      {/* Floating button */}
      <button
        onClick={() => setOpen(prev => !prev)}
        className="w-14 h-14 rounded-full flex items-center justify-center text-xl text-white transition-all duration-200 hover:scale-105"
        style={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
          boxShadow: '0 4px 20px rgba(99,102,241,0.4)',
          border: 'none',
          cursor: 'pointer',
          zIndex: 1001,
        }}
        onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 6px 28px rgba(99,102,241,0.6)' }}
        onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 4px 20px rgba(99,102,241,0.4)' }}
        title="Apri CineBot"
      >
        {open ? '✕' : '🎬'}
        {!open && unread > 0 && (
          <span style={{
            position: 'absolute',
            top: 0,
            right: 0,
            background: '#ef4444',
            color: '#fff',
            borderRadius: '50%',
            width: 18,
            height: 18,
            fontSize: '0.65rem',
            fontWeight: 700,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            {unread}
          </span>
        )}
      </button>

      <style>{`
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  )
}
