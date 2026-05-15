import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

const FALLBACK = 'https://via.placeholder.com/150x225/1f2937/6b7280?text=?'

const tabActive = {
  background: 'rgba(129,140,248,0.15)',
  border: '1px solid rgba(129,140,248,0.3)',
  color: '#a5b4fc',
}
const tabInactive = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid rgba(255,255,255,0.08)',
  color: '#94a3b8',
}

export default function Profile() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [tab, setTab] = useState('favorites')
  const [favorites, setFavorites] = useState([])
  const [ratings, setRatings] = useState([])
  const [favoriteSet, setFavoriteSet] = useState(new Set())
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.get('/profile/favorites'),
      api.get('/profile/ratings'),
    ]).then(([favRes, ratRes]) => {
      const favs = favRes.data.favorites ?? []
      setFavorites(favs)
      setFavoriteSet(new Set(favs.map(f => f.id)))
      setRatings(ratRes.data.ratings ?? [])
    }).finally(() => setLoading(false))
  }, [])

  async function handleFavorite(movieId) {
    const res = await api.post(`/movies/${movieId}/favorite`)
    if (!res.data.is_favorite) {
      setFavorites(prev => prev.filter(f => f.id !== movieId))
      setFavoriteSet(prev => { const s = new Set(prev); s.delete(movieId); return s })
    }
  }

  const initials = (user?.username ?? 'U').slice(0, 2).toUpperCase()
  const avgRating = ratings.length
    ? (ratings.reduce((s, r) => s + r.rating, 0) / ratings.length).toFixed(1)
    : null

  return (
    <main className="pt-6 pb-12 px-4 max-w-3xl mx-auto">
      {/* Card profilo */}
      <div
        className="rounded-2xl p-6 mb-6"
        style={{
          background: 'rgba(255,255,255,0.05)',
          backdropFilter: 'blur(12px)',
          WebkitBackdropFilter: 'blur(12px)',
          border: '1px solid rgba(255,255,255,0.10)',
        }}
      >
        <div className="flex items-center gap-5">
          <div
            className="w-16 h-16 rounded-full flex items-center justify-center text-white text-2xl font-bold flex-shrink-0"
            style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
          >
            {initials}
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100">{user?.username}</h1>
            <p className="text-slate-400 text-sm">{user?.email}</p>
            <p className="text-slate-500 text-sm mt-1">
              {ratings.length} film valutati · {favorites.length} preferiti
              {avgRating && ` · ${avgRating} ★ media`}
            </p>
          </div>
        </div>
      </div>

      {/* Tab selector */}
      <div
        className="flex gap-1 p-1 rounded-xl mb-6 w-fit"
        style={{
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid rgba(255,255,255,0.08)',
        }}
      >
        {[['favorites', '❤️ Preferiti'], ['rated', '★ Valutati']].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
            style={tab === key ? tabActive : tabInactive}
          >
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-slate-400 text-center py-12 text-2xl animate-spin">⟳</div>
      ) : tab === 'favorites' ? (
        favorites.length === 0 ? (
          <EmptyState icon="🎬" text="Nessun film nei preferiti" cta="Esplora il catalogo" onCta={() => navigate('/movies')} />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
            {favorites.map(m => (
              <MovieCard
                key={m.id}
                movie={m}
                isFavorite={favoriteSet.has(m.id)}
                onFavorite={handleFavorite}
                showRating={false}
              />
            ))}
          </div>
        )
      ) : (
        ratings.length === 0 ? (
          <EmptyState icon="⭐" text="Nessun film valutato" cta="Vai al catalogo" onCta={() => navigate('/movies')} />
        ) : (
          <div className="flex flex-col gap-2">
            {ratings.map(r => (
              <div
                key={r.movie_id}
                className="flex items-center gap-4 rounded-xl p-3 cursor-pointer transition-all duration-200"
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.07)'}
                onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
                onClick={() => navigate(`/movies/${r.movie_id}`)}
              >
                <img
                  src={r.poster_url || FALLBACK}
                  alt={r.title}
                  className="w-10 h-14 object-cover rounded-lg flex-shrink-0"
                  onError={e => { e.target.src = FALLBACK }}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-slate-100 text-sm font-medium truncate">{r.title}</p>
                  <p className="text-slate-500 text-xs">{r.year} · {r.genres?.split('|').slice(0, 2).join(', ')}</p>
                </div>
                <div className="text-amber-400 text-sm font-semibold whitespace-nowrap">
                  {'★'.repeat(Math.round(r.rating))} {r.rating}
                </div>
              </div>
            ))}
          </div>
        )
      )}
    </main>
  )
}

function EmptyState({ icon, text, cta, onCta }) {
  return (
    <div className="flex flex-col items-center gap-4 py-16 text-center">
      <span className="text-5xl">{icon}</span>
      <p className="text-slate-400">{text}</p>
      <button
        onClick={onCta}
        className="text-white px-5 py-2 rounded-xl text-sm transition-all"
        style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
      >
        {cta}
      </button>
    </div>
  )
}
