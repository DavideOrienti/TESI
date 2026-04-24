import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import api from '../api/client'
import StarRating from '../components/StarRating'
import MovieCard from '../components/MovieCard'

const FALLBACK = 'https://via.placeholder.com/300x450/1f2937/6b7280?text=No+Poster'

export default function MovieDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [userRating, setUserRating] = useState(null)
  const [isFavorite, setIsFavorite] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    api.get(`/movies/${id}`)
      .then(res => {
        setData(res.data)
        setUserRating(res.data.user_rating)
        setIsFavorite(res.data.is_favorite)
      })
      .catch(err => setError(err.response?.status === 404 ? 'Film non trovato' : 'Errore di rete'))
      .finally(() => setLoading(false))
  }, [id])

  async function handleRate(v) {
    await api.post(`/movies/${id}/rate`, { rating: v })
    setUserRating(v)
  }

  async function handleFavorite() {
    const res = await api.post(`/movies/${id}/favorite`)
    setIsFavorite(res.data.is_favorite)
  }

  if (loading) return <div className="flex justify-center py-20 text-gray-400 text-3xl animate-spin">⟳</div>
  if (error) return (
    <div className="flex flex-col items-center py-20 gap-4">
      <p className="text-red-400 text-lg">{error}</p>
      <button onClick={() => navigate(-1)} className="text-indigo-400 hover:text-indigo-300">← Torna indietro</button>
    </div>
  )

  const { movie, similar_movies } = data
  const genres = movie.genres ? movie.genres.split('|') : []
  const actors = movie.actors_top5 ? movie.actors_top5.split(',').map(s => s.trim()) : []

  return (
    <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <button onClick={() => navigate(-1)} className="text-gray-400 hover:text-white mb-6 flex items-center gap-1">
        ← Indietro
      </button>

      {/* Layout principale */}
      <div className="flex flex-col md:flex-row gap-8 mb-10">
        {/* Sinistra: poster + azioni */}
        <div className="flex-shrink-0 flex flex-col items-center gap-4">
          <img
            src={movie.poster_url || FALLBACK}
            alt={movie.title}
            className="w-56 rounded-xl shadow-2xl"
            onError={e => { e.target.src = FALLBACK }}
          />

          <div className="text-center">
            <p className="text-gray-400 text-sm mb-2">
              {userRating ? `Hai valutato: ${userRating} ★` : 'Dai un voto'}
            </p>
            <StarRating value={userRating ?? 0} onChange={handleRate} size="lg" />
          </div>

          <button
            onClick={handleFavorite}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              isFavorite
                ? 'bg-red-900/40 text-red-400 border border-red-800 hover:bg-red-900/60'
                : 'bg-gray-700 text-gray-300 border border-gray-600 hover:bg-gray-600'
            }`}
          >
            {isFavorite ? '❤️ Nei preferiti' : '🤍 Aggiungi ai preferiti'}
          </button>
        </div>

        {/* Destra: info */}
        <div className="flex flex-col gap-4 flex-1">
          <div>
            <h1 className="text-3xl font-bold text-white leading-tight">
              {movie.title_clean || movie.title}
            </h1>
            <p className="text-gray-400 text-lg mt-1">{movie.year}</p>
          </div>

          {/* Generi */}
          <div className="flex flex-wrap gap-2">
            {genres.map(g => (
              <span key={g} className="bg-indigo-900/50 text-indigo-300 border border-indigo-800 px-2.5 py-0.5 rounded-full text-sm">
                {g}
              </span>
            ))}
          </div>

          {movie.director && (
            <p className="text-gray-300">
              <span className="text-gray-500">Regia di </span>{movie.director}
            </p>
          )}

          {actors.length > 0 && (
            <p className="text-gray-300">
              <span className="text-gray-500">Con </span>{actors.join(', ')}
            </p>
          )}

          {movie.overview_en && (
            <p className="text-gray-400 leading-relaxed text-sm mt-2">{movie.overview_en}</p>
          )}
        </div>
      </div>

      {/* Film simili */}
      {similar_movies?.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold text-white mb-4">Film simili</h2>
          <div className="flex gap-4 overflow-x-auto pb-2">
            {similar_movies.map(s => {
              const m = data.__simMovies?.[s.movie_id]
              return (
                <div key={s.movie_id} className="flex-shrink-0 w-32">
                  <div
                    className="bg-gray-800 rounded-lg overflow-hidden cursor-pointer hover:scale-105 transition-transform"
                    onClick={() => navigate(`/movies/${s.movie_id}`)}
                  >
                    <SimilarCard movieId={s.movie_id} similarity={s.similarity} />
                  </div>
                </div>
              )
            })}
          </div>
        </section>
      )}
    </main>
  )
}

function SimilarCard({ movieId, similarity }) {
  const [movie, setMovie] = useState(null)
  const FALLBACK = 'https://via.placeholder.com/150x225/1f2937/6b7280?text=?'

  useEffect(() => {
    api.get(`/movies/${movieId}`).then(res => setMovie(res.data.movie)).catch(() => {})
  }, [movieId])

  if (!movie) return (
    <div className="aspect-[2/3] bg-gray-700 animate-pulse rounded" />
  )

  return (
    <div>
      <img
        src={movie.poster_url || FALLBACK}
        alt={movie.title}
        className="w-full aspect-[2/3] object-cover rounded-t"
        onError={e => { e.target.src = FALLBACK }}
      />
      <div className="p-2">
        <p className="text-xs text-white line-clamp-2 leading-tight">{movie.title_clean || movie.title}</p>
        <p className="text-xs text-gray-500 mt-0.5">{Math.round(similarity * 100)}% sim.</p>
      </div>
    </div>
  )
}
