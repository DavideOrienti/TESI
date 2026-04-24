import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()
  const [form, setForm] = useState({ username: '', email: '', password: '', confirm: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

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
    <div>
      <label className="block text-sm text-gray-300 mb-1">{label}</label>
      <input
        type={type}
        value={form[key]}
        onChange={e => setForm(f => ({ ...f, [key]: e.target.value }))}
        required
        className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
        placeholder={placeholder}
      />
    </div>
  )

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900 px-4">
      <div className="w-full max-w-sm bg-gray-800 rounded-xl p-8 shadow-2xl">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white">CineRec</h1>
          <p className="text-gray-400 text-sm mt-1">Crea il tuo account</p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {field('username', 'Username', 'text', 'tuonome')}
          {field('email', 'Email', 'email', 'tu@email.com')}
          {field('password', 'Password', 'password', '••••••••')}
          {field('confirm', 'Conferma password', 'password', '••••••••')}

          {error && <p className="text-red-400 text-sm text-center">{error}</p>}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium rounded-lg py-2.5 transition-colors"
          >
            {loading ? 'Registrazione...' : 'Registrati'}
          </button>
        </form>

        <p className="text-center text-gray-400 text-sm mt-6">
          Hai già un account?{' '}
          <Link to="/login" className="text-indigo-400 hover:text-indigo-300">Accedi</Link>
        </p>
      </div>
    </div>
  )
}
