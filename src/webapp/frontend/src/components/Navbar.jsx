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

  return (
    <nav className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link to="/home" className="text-indigo-400 font-bold text-xl tracking-tight">
            CineRec
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-6">
            {links.map(l => (
              <Link
                key={l.to}
                to={l.to}
                className={`text-sm font-medium transition-colors ${
                  isActive(l.to) ? 'text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                {l.label}
              </Link>
            ))}
          </div>

          {/* User + logout */}
          <div className="hidden md:flex items-center gap-3">
            <span className="text-gray-400 text-sm">{user?.username}</span>
            <button
              onClick={logout}
              className="text-sm text-gray-400 hover:text-white border border-gray-700 rounded px-3 py-1 transition-colors"
            >
              Esci
            </button>
          </div>

          {/* Mobile hamburger */}
          <button
            className="md:hidden text-gray-400 hover:text-white"
            onClick={() => setMenuOpen(o => !o)}
          >
            <span className="text-2xl">{menuOpen ? '✕' : '☰'}</span>
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden bg-gray-900 border-t border-gray-800 px-4 py-3 flex flex-col gap-3">
          {links.map(l => (
            <Link
              key={l.to}
              to={l.to}
              onClick={() => setMenuOpen(false)}
              className={`text-sm font-medium ${isActive(l.to) ? 'text-white' : 'text-gray-400'}`}
            >
              {l.label}
            </Link>
          ))}
          <div className="border-t border-gray-800 pt-3 flex items-center justify-between">
            <span className="text-gray-400 text-sm">{user?.username}</span>
            <button onClick={logout} className="text-sm text-red-400">Esci</button>
          </div>
        </div>
      )}
    </nav>
  )
}
