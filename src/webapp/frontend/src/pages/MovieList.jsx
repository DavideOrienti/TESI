import { useEffect, useState, useCallback } from 'react'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

const GENRES = [
  '', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]

export default function MovieList() {
  const [tab, setTab] = useState('title') // 'title' | 'semantic'

  // --- title search state ---
  const [movies, setMovies] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [genre, setGenre] = useState('')

  // --- semantic search state ---
  const [semanticInput, setSemanticInput] = useState('')
  const [semanticResults, setSemanticResults] = useState([])
  const [semanticQuery, setSemanticQuery] = useState('')
  const [semanticLoading, setSemanticLoading] = useState(false)
  const [semanticError, setSemanticError] = useState('')

  // --- shared ---
  const [userRatings, setUserRatings] = useState({})
  const [favorites, setFavorites] = useState(new Set())
  const [loading, setLoading] = useState(true)
  const LIMIT = 20

  const loadMovies = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ page, limit: LIMIT })
      if (search) params.set('search', search)
      if (genre) params.set('genre', genre)
      const res = await api.get(`/movies?${params}`)
      setMovies(res.data.movies)
      setTotal(res.data.total)
    } finally {
      setLoading(false)
    }
  }, [page, search, genre])

  useEffect(() => { if (tab === 'title') loadMovies() }, [loadMovies, tab])

  useEffect(() => {
    api.get('/profile/ratings').then(res => {
      const m = {}
      for (const r of res.data.ratings ?? []) m[r.movie_id] = r.rating
      setUserRatings(m)
    }).catch(() => {})
    api.get('/profile/favorites').then(res => {
      setFavorites(new Set((res.data.favorites ?? []).map(f => f.id)))
    }).catch(() => {})
  }, [])

  function handleSearch(e) {
    e.preventDefault()
    setSearch(searchInput)
    setPage(1)
  }

  function handleGenre(g) {
    setGenre(g)
    setPage(1)
  }

  async function handleSemanticSearch(e) {
    e.preventDefault()
    if (semanticInput.trim().length < 3) return
    setSemanticLoading(true)
    setSemanticError('')
    setSemanticResults([])
    setSemanticQuery(semanticInput.trim())
    try {
      const res = await api.get(`/search/semantic?q=${encodeURIComponent(semanticInput.trim())}&top_k=20`)
      setSemanticResults(res.data.results ?? [])
    } catch (err) {
      if (err.response?.status === 503) {
        setSemanticError('Ricerca semantica non disponibile al momento.')
      } else {
        setSemanticError('Errore durante la ricerca. Riprova.')
      }
    } finally {
      setSemanticLoading(false)
    }
  }

  async function handleRate(movieId, rating) {
    await api.post(`/movies/${movieId}/rate`, { rating })
    setUserRatings(prev => ({ ...prev, [movieId]: rating }))
  }

  async function handleFavorite(movieId) {
    const res = await api.post(`/movies/${movieId}/favorite`)
    setFavorites(prev => {
      const next = new Set(prev)
      res.data.is_favorite ? next.add(movieId) : next.delete(movieId)
      return next
    })
  }

  const totalPages = Math.ceil(total / LIMIT)

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 className="text-2xl font-bold text-white mb-6">Catalogo Film</h1>

      {/* Tab switcher */}
      <div className="flex gap-1 bg-gray-800 rounded-lg p-1 mb-6 w-fit">
        <button
          onClick={() => setTab('title')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            tab === 'title' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          🔍 Titolo
        </button>
        <button
          onClick={() => setTab('semantic')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            tab === 'semantic' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          ✨ Concetto
        </button>
      </div>

      {tab === 'title' ? (
        <>
          {/* Ricerca per titolo + filtro genere */}
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <form onSubmit={handleSearch} className="flex gap-2 flex-1">
              <input
                type="text"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                placeholder="Cerca per titolo..."
                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
              />
              <button
                type="submit"
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg transition-colors"
              >
                Cerca
              </button>
            </form>
            <select
              value={genre}
              onChange={e => handleGenre(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-gray-300 focus:outline-none focus:border-indigo-500"
            >
              <option value="">Tutti i generi</option>
              {GENRES.filter(Boolean).map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>

          {loading ? (
            <div className="flex justify-center py-16 text-gray-400 text-2xl animate-spin">⟳</div>
          ) : (
            <>
              <p className="text-gray-400 text-sm mb-4">{total} film trovati</p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 mb-8">
                {movies.map(m => (
                  <MovieCard
                    key={m.id}
                    movie={m}
                    userRating={userRatings[m.id]}
                    isFavorite={favorites.has(m.id)}
                    onRate={handleRate}
                    onFavorite={handleFavorite}
                  />
                ))}
              </div>
              {totalPages > 1 && (
                <div className="flex items-center justify-center gap-4">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white rounded-lg transition-colors"
                  >
                    ← Precedente
                  </button>
                  <span className="text-gray-400 text-sm">Pagina {page} di {totalPages}</span>
                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-white rounded-lg transition-colors"
                  >
                    Successiva →
                  </button>
                </div>
              )}
            </>
          )}
        </>
      ) : (
        <>
          {/* Ricerca semantica */}
          <form onSubmit={handleSemanticSearch} className="flex gap-2 mb-2">
            <input
              type="text"
              value={semanticInput}
              onChange={e => setSemanticInput(e.target.value)}
              placeholder="es. film con maghi anni 70, thriller psicologico, storia d'amore a Parigi..."
              className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
            />
            <button
              type="submit"
              disabled={semanticLoading || semanticInput.trim().length < 3}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-4 py-2 rounded-lg transition-colors whitespace-nowrap"
            >
              {semanticLoading ? '⟳' : '✨ Cerca'}
            </button>
          </form>
          <p className="text-xs text-gray-500 mb-6">
            Cerca per trama, ambientazione, periodo storico, mood o qualsiasi concetto — non solo titolo
          </p>

          {semanticError && (
            <p className="text-red-400 text-sm mb-4">{semanticError}</p>
          )}

          {semanticResults.length > 0 && (
            <>
              <p className="text-gray-400 text-sm mb-4">
                {semanticResults.length} risultati per "{semanticQuery}"
              </p>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
                {semanticResults.map(m => (
                  <div key={m.movie_id} className="flex flex-col">
                    <MovieCard
                      movie={{ ...m, id: m.movie_id }}
                      userRating={userRatings[m.movie_id]}
                      isFavorite={favorites.has(m.movie_id)}
                      onRate={handleRate}
                      onFavorite={handleFavorite}
                    />
                    <div className="mt-1 px-1">
                      <span className="text-xs bg-indigo-900/50 text-indigo-300 px-2 py-0.5 rounded-full">
                        Pertinenza: {Math.round(m.similarity_score * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {!semanticLoading && semanticQuery && semanticResults.length === 0 && !semanticError && (
            <p className="text-gray-400 text-center py-12">Nessun risultato trovato per "{semanticQuery}"</p>
          )}
        </>
      )}
    </main>
  )
}
