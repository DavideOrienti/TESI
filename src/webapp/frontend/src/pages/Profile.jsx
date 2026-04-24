import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

const FALLBACK = 'https://via.placeholder.com/150x225/1f2937/6b7280?text=?'

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

  const initial = (user?.username ?? 'U')[0].toUpperCase()
  const colors = ['bg-indigo-600', 'bg-purple-600', 'bg-teal-600', 'bg-pink-600']
  const color = colors[initial.charCodeAt(0) % colors.length]

  return (
    <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="flex items-center gap-5 mb-8">
        <div className={`${color} w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold text-white shadow-lg`}>
          {initial}
        </div>
        <div>
          <h1 className="text-2xl font-bold text-white">{user?.username}</h1>
          <p className="text-gray-400 text-sm">{user?.email}</p>
          <p className="text-gray-500 text-xs mt-1">
            {ratings.length} film valutati · {favorites.length} preferiti
          </p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-gray-800 rounded-lg p-1 mb-6 w-fit">
        {[['favorites', '❤️ Preferiti'], ['rated', '★ Valutati']].map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              tab === key ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-gray-400 text-center py-12 text-2xl animate-spin">⟳</div>
      ) : tab === 'favorites' ? (
        favorites.length === 0 ? (
          <EmptyState
            icon="🎬"
            text="Nessun film nei preferiti"
            cta="Esplora il catalogo"
            onCta={() => navigate('/movies')}
          />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
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
          <EmptyState
            icon="⭐"
            text="Nessun film valutato"
            cta="Vai al catalogo"
            onCta={() => navigate('/movies')}
          />
        ) : (
          <div className="flex flex-col gap-3">
            {ratings.map(r => (
              <div
                key={r.movie_id}
                className="flex items-center gap-4 bg-gray-800 rounded-lg p-3 hover:bg-gray-750 cursor-pointer transition-colors"
                onClick={() => navigate(`/movies/${r.movie_id}`)}
              >
                <img
                  src={r.poster_url || FALLBACK}
                  alt={r.title}
                  className="w-10 h-14 object-cover rounded flex-shrink-0"
                  onError={e => { e.target.src = FALLBACK }}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm font-medium truncate">{r.title}</p>
                  <p className="text-gray-500 text-xs">{r.year} · {r.genres?.split('|').slice(0, 2).join(', ')}</p>
                </div>
                <div className="text-yellow-400 text-sm font-semibold whitespace-nowrap">
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
      <p className="text-gray-400">{text}</p>
      <button
        onClick={onCta}
        className="bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2 rounded-lg text-sm transition-colors"
      >
        {cta}
      </button>
    </div>
  )
}
