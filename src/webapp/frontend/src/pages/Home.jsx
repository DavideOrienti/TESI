import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

function HorizontalScroll({ title, subtitle, movies, userRatings, favorites, onRate, onFavorite }) {
  return (
    <section className="mb-10">
      <h2 className="text-lg font-semibold text-white mb-1">{title}</h2>
      {subtitle && <p className="text-xs text-gray-500 mb-3">{subtitle}</p>}
      <div className="flex gap-4 overflow-x-auto pb-2">
        {movies.map(m => (
          <div key={m.movie_id ?? m.id} className="flex-shrink-0 w-36">
            <MovieCard
              movie={{ ...m, id: m.movie_id ?? m.id }}
              userRating={userRatings[m.movie_id ?? m.id]}
              isFavorite={favorites.has(m.movie_id ?? m.id)}
              onRate={onRate}
              onFavorite={onFavorite}
              explanation={m.explanation}
            />
          </div>
        ))}
      </div>
    </section>
  )
}

function GridSection({ title, movies, userRatings, favorites, onRate, onFavorite }) {
  return (
    <section className="mb-10">
      <h2 className="text-lg font-semibold text-white mb-4">{title}</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {movies.map(m => (
          <MovieCard
            key={m.movie_id ?? m.id}
            movie={{ ...m, id: m.movie_id ?? m.id }}
            userRating={userRatings[m.movie_id ?? m.id]}
            isFavorite={favorites.has(m.movie_id ?? m.id)}
            onRate={onRate}
            onFavorite={onFavorite}
            explanation={m.explanation}
          />
        ))}
      </div>
    </section>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [recs, setRecs] = useState([])
  const [social, setSocial] = useState([])
  const [popular, setPopular] = useState([])
  const [userRatings, setUserRatings] = useState({})
  const [favorites, setFavorites] = useState(new Set())
  const [loading, setLoading] = useState(true)
  const [ratingCount, setRatingCount] = useState(0)

  const loadAll = useCallback(async () => {
    setLoading(true)
    try {
      const [recRes, popRes, socialRes, ratingsRes, favRes] = await Promise.all([
        api.get('/recommendations?top_k=20').catch(() => ({ data: { recommendations: [] } })),
        api.get('/recommendations/popular?top_k=20'),
        api.get('/recommendations/social?top_k=10').catch(() => ({ data: { recommendations: [], has_enough_data: false } })),
        api.get('/profile/ratings').catch(() => ({ data: { ratings: [] } })),
        api.get('/profile/favorites').catch(() => ({ data: { favorites: [] } })),
      ])

      setRecs(recRes.data.recommendations ?? [])
      setPopular(popRes.data.movies ?? [])
      setSocial(socialRes.data.has_enough_data ? (socialRes.data.recommendations ?? []) : [])

      const ratingsMap = {}
      for (const r of ratingsRes.data.ratings ?? []) ratingsMap[r.movie_id] = r.rating
      setUserRatings(ratingsMap)
      setRatingCount(Object.keys(ratingsMap).length)

      setFavorites(new Set((favRes.data.favorites ?? []).map(f => f.id)))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadAll() }, [loadAll])

  async function handleRate(movieId, rating) {
    await api.post(`/movies/${movieId}/rate`, { rating })
    setUserRatings(prev => ({ ...prev, [movieId]: rating }))
    setRatingCount(prev => prev + (userRatings[movieId] ? 0 : 1))
  }

  async function handleFavorite(movieId) {
    const res = await api.post(`/movies/${movieId}/favorite`)
    setFavorites(prev => {
      const next = new Set(prev)
      res.data.is_favorite ? next.add(movieId) : next.delete(movieId)
      return next
    })
  }

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-gray-400">
      <div className="animate-spin text-4xl">⟳</div>
    </div>
  )

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

      {/* Prompt onboarding */}
      {ratingCount < 5 && (
        <div className="bg-indigo-900/40 border border-indigo-700 rounded-xl p-6 mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h3 className="text-white font-semibold">Personalizza le tue raccomandazioni</h3>
            <p className="text-gray-400 text-sm mt-1">
              Valuta almeno {5 - ratingCount} film per ricevere consigli personalizzati.
            </p>
          </div>
          <button
            onClick={() => navigate('/movies')}
            className="bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors whitespace-nowrap"
          >
            Vai al catalogo
          </button>
        </div>
      )}

      {/* Raccomandazioni personalizzate (hybrid) */}
      {recs.length > 0 && (
        <GridSection
          title="Consigliati per te"
          movies={recs}
          userRatings={userRatings}
          favorites={favorites}
          onRate={handleRate}
          onFavorite={handleFavorite}
        />
      )}

      {/* Social CF — appare solo se has_enough_data=true */}
      {social.length > 0 && (
        <HorizontalScroll
          title="👥 Piace a utenti con gusti simili ai tuoi"
          subtitle="Basato sui film apprezzati da utenti con preferenze simili alle tue"
          movies={social}
          userRatings={userRatings}
          favorites={favorites}
          onRate={handleRate}
          onFavorite={handleFavorite}
        />
      )}

      {/* Popolari */}
      {popular.length > 0 && (
        <HorizontalScroll
          title="I più popolari in questo momento"
          movies={popular}
          userRatings={userRatings}
          favorites={favorites}
          onRate={handleRate}
          onFavorite={handleFavorite}
        />
      )}
    </main>
  )
}
