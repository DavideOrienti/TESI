import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const inputStyle = (focused) => ({
  background: 'rgba(255,255,255,0.06)',
  border: `1px solid ${focused ? 'rgba(129,140,248,0.5)' : 'rgba(255,255,255,0.10)'}`,
  transition: 'border-color 0.2s',
})

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', email: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [focused, setFocused] = useState({})

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    if (form.password !== form.confirm) { setError('Le password non coincidono'); return }
    if (form.password.length < 6) { setError('La password deve avere almeno 6 caratteri'); return }
    setLoading(true)
    try {
      await register(form.username, form.email, form.password)
      navigate('/home')
    } catch (err) {
      setError(err.response?.data?.error || 'Errore durante la registrazione')
    } finally {
      setLoading(false)
    }
  }

  const field = (key, label, type = 'text', placeholder = '') => (
    <div key={key}>
      <label className="block text-xs text-slate-400 uppercase tracking-wide mb-1.5">
        {label}
      </label>
      <input
        type={type}
        value={form[key]}
        onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
        onFocus={() => setFocused(f => ({ ...f, [key]: true }))}
        onBlur={() => setFocused(f => ({ ...f, [key]: false }))}
        required
        className="w-full rounded-xl px-4 py-3 text-slate-100 text-sm placeholder:text-slate-600 outline-none"
        style={inputStyle(focused[key])}
        placeholder={placeholder}
      />
    </div>
  )

  return (
    <div className="min-h-screen flex items-center justify-center px-4 py-8">
      <div
        className="w-full max-w-sm rounded-3xl p-8"
        style={{
          background: 'rgba(255,255,255,0.07)',
          backdropFilter: 'blur(20px)',
          WebkitBackdropFilter: 'blur(20px)',
          border: '1px solid rgba(255,255,255,0.12)',
          boxShadow: '0 25px 50px rgba(0,0,0,0.5)',
        }}
      >
        <div className="text-center mb-6">
          <div className="text-3xl mb-2">🎬</div>
          <div className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent">
            CineRec
          </div>
          <p className="text-slate-400 text-sm mt-1">Crea il tuo account</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {field('username', 'Username', 'text', 'tuonome')}
          {field('email', 'Email', 'email', 'tu@email.com')}
          {field('password', 'Password', 'password', '••••••••')}
          {field('confirm', 'Conferma password', 'password', '••••••••')}

          {error && <p className="text-red-400 text-sm text-center">{error}</p>}

          <SubmitButton loading={loading} label="Registrati" loadingLabel="Registrazione..." />
        </form>

        <p className="text-center text-slate-400 text-sm mt-6">
          Hai già un account?{' '}
          <Link to="/login" className="text-indigo-400 hover:text-indigo-300 transition-colors duration-150">
            Accedi
          </Link>
        </p>
      </div>
    </div>
  )
}

function SubmitButton({ loading, label, loadingLabel }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      type="submit"
      disabled={loading}
      className="w-full py-3 rounded-xl font-medium text-sm text-white transition-all duration-200 mt-6 disabled:opacity-50"
      style={{
        background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
        boxShadow: hovered
          ? '0 8px 25px rgba(99,102,241,0.5)'
          : '0 4px 15px rgba(99,102,241,0.3)',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {loading ? loadingLabel : label}
    </button>
  )
}
