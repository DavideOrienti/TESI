import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

export default function Navbar() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)

  const links = [
    { to: '/home', label: 'Home' },
    { to: '/movies', label: 'Catalogo' },
    { to: '/social', label: 'Grafo' },
    { to: '/profile', label: 'Profilo' },
  ]

  const isActive = (path) => location.pathname === path

  const initials = user?.username
    ? user.username.slice(0, 2).toUpperCase()
    : '??'

  return (
    <nav className="glass fixed top-0 left-0 right-0 z-50 border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link
            to="/home"
            className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent"
          >
            CineRec
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-6">
            {links.map(l => (
              <Link
                key={l.to}
                to={l.to}
                className={`text-sm font-medium transition-colors duration-200 ${
                  isActive(l.to)
                    ? 'text-indigo-400'
                    : 'text-slate-300 hover:text-indigo-400'
                }`}
              >
                {l.label}
              </Link>
            ))}
          </div>

          {/* User + logout */}
          <div className="hidden md:flex items-center gap-3">
            <div className="flex items-center gap-2">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-indigo-300"
                style={{
                  background: 'rgba(99,102,241,0.2)',
                  border: '1px solid rgba(99,102,241,0.3)',
                }}
              >
                {initials}
              </div>
              <span className="text-slate-300 text-sm">{user?.username}</span>
            </div>
            <button
              onClick={logout}
              className="text-sm text-rose-400 hover:text-rose-300 transition-colors duration-200 ml-1"
            >
              Esci
            </button>
          </div>

          {/* Mobile hamburger */}
          <button
            className="md:hidden text-slate-400 hover:text-white"
            onClick={() => setMenuOpen(o => !o)}
          >
            <span className="text-2xl">{menuOpen ? '✕' : '☰'}</span>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden glass border-t border-white/10 px-4 py-3 flex flex-col gap-3">
          {links.map(l => (
            <Link
              key={l.to}
              to={l.to}
              onClick={() => setMenuOpen(false)}
              className={`text-sm font-medium transition-colors duration-200 ${
                isActive(l.to) ? 'text-indigo-400' : 'text-slate-300'
              }`}
            >
              {l.label}
            </Link>
          ))}
          <div className="border-t border-white/10 pt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-indigo-300"
                style={{
                  background: 'rgba(99,102,241,0.2)',
                  border: '1px solid rgba(99,102,241,0.3)',
                }}
              >
                {initials}
              </div>
              <span className="text-slate-300 text-sm">{user?.username}</span>
            </div>
            <button
              onClick={logout}
              className="text-sm text-rose-400 hover:text-rose-300 transition-colors duration-200"
            >
              Esci
            </button>
          </div>
        </div>
      )}
    </nav>
  )
}
