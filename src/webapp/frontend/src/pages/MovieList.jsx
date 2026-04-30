import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

const GENRES = [
  '', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]

export default function MovieList() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('title') // 'title' | 'semantic' | 'visual'

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

  // --- visual search state ---
  const [visualFile, setVisualFile] = useState(null)
  const [visualPreview, setVisualPreview] = useState(null)
  const [visualRecognized, setVisualRecognized] = useState(null)
  const [visualSimilar, setVisualSimilar] = useState([])
  const [visualContentSimilar, setVisualContentSimilar] = useState([])
  const [visualLoading, setVisualLoading] = useState(false)
  const [visualError, setVisualError] = useState('')
  const [visualNotFound, setVisualNotFound] = useState(false)

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

  function handleVisualFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setVisualFile(file)
    setVisualRecognized(null)
    setVisualSimilar([])
    setVisualContentSimilar([])
    setVisualError('')
    setVisualNotFound(false)
    const reader = new FileReader()
    reader.onload = ev => setVisualPreview(ev.target.result)
    reader.readAsDataURL(file)
  }

  async function handleVisualSearch() {
    if (!visualFile) return
    setVisualLoading(true)
    setVisualError('')
    setVisualRecognized(null)
    setVisualSimilar([])
    setVisualContentSimilar([])
    setVisualNotFound(false)
    try {
      const reader = new FileReader()
      const base64 = await new Promise((resolve, reject) => {
        reader.onload = ev => resolve(ev.target.result.split(',')[1])
        reader.onerror = reject
        reader.readAsDataURL(visualFile)
      })
      const res = await api.post('/visual/search', { image: base64, top_k: 5 })
      const recognized = res.data.recognized_movie ?? null
      setVisualRecognized(recognized)
      setVisualSimilar(res.data.similar_movies ?? [])
      // Fetch film simili per trama dal riconosciuto
      if (recognized?.movie_id) {
        api.get(`/movies/${recognized.movie_id}`)
          .then(r => setVisualContentSimilar(r.data.similar_movies?.slice(0, 6) ?? []))
          .catch(() => {})
      }
    } catch (err) {
      if (err.response?.status === 404) {
        setVisualNotFound(true)
      } else if (err.response?.status === 503) {
        setVisualError('Ricerca per immagine non disponibile in questo momento.')
      } else {
        setVisualError('Errore durante la ricerca. Riprova.')
      }
    } finally {
      setVisualLoading(false)
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
        <button
          onClick={() => setTab('visual')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            tab === 'visual' ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:text-white'
          }`}
        >
          🖼️ Foto locandina
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

      {tab === 'visual' && (
        <>
          {/* Upload */}
          <div className="mb-6">
            <label className="flex items-center gap-3 cursor-pointer w-fit">
              <span className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                📁 Carica foto locandina
              </span>
              <input type="file" accept="image/*" className="hidden" onChange={handleVisualFileChange} />
            </label>
            <p className="text-xs text-gray-500 mt-2">
              Carica una foto di una locandina per trovare il film o film con poster simili
            </p>
          </div>

          {visualPreview && (
            <div className="flex items-start gap-6 mb-6">
              <img
                src={visualPreview}
                alt="preview"
                className="w-24 h-36 object-cover rounded-lg border border-gray-600"
              />
              <div className="flex flex-col gap-2">
                <p className="text-gray-400 text-sm">{visualFile?.name}</p>
                <button
                  onClick={handleVisualSearch}
                  disabled={visualLoading}
                  className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-4 py-2 rounded-lg text-sm transition-colors w-fit"
                >
                  {visualLoading ? '⟳ Analisi in corso...' : '🖼️ Riconosci film'}
                </button>
              </div>
            </div>
          )}

          {visualError && <p className="text-red-400 text-sm mb-4">{visualError}</p>}

          {visualNotFound && (
            <p className="text-yellow-400 text-sm mb-4">
              Film non riconosciuto — prova con una foto più nitida o un'angolazione diversa.
            </p>
          )}

          {/* Sezione 1 — Film riconosciuto */}
          {visualRecognized && (
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-white mb-4">Film riconosciuto</h2>
              <VisualRecognizedCard
                movie={visualRecognized}
                navigate={navigate}
                userRatings={userRatings}
                favorites={favorites}
                onRate={handleRate}
                onFavorite={handleFavorite}
              />
            </div>
          )}

          {/* Sezione 2 — Film con poster simili (riga scrollabile, max 5) */}
          {visualSimilar.length > 0 && (
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-white mb-4">Film con poster simili</h2>
              <div className="flex gap-4 overflow-x-auto pb-2">
                {visualSimilar.slice(0, 5).map(m => (
                  <div key={m.movie_id} className="flex-shrink-0 w-36 flex flex-col">
                    <MovieCard
                      movie={{ ...m, id: m.movie_id }}
                      userRating={userRatings[m.movie_id]}
                      isFavorite={favorites.has(m.movie_id)}
                      onRate={handleRate}
                      onFavorite={handleFavorite}
                    />
                    <div className="mt-1 px-1">
                      <span className="text-xs bg-purple-900/50 text-purple-300 px-2 py-0.5 rounded-full">
                        Somiglianza visiva: {Math.round(m.visual_similarity * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Sezione 3 — Ti potrebbe piacere anche (simili per trama) */}
          {visualContentSimilar.length > 0 && visualRecognized && (
            <div className="mb-8">
              <h2 className="text-lg font-semibold text-white mb-4">
                Ti potrebbe piacere anche —{' '}
                <span className="text-gray-400 font-normal">basato sulla trama di {visualRecognized.title}</span>
              </h2>
              <div className="grid grid-cols-3 sm:grid-cols-3 md:grid-cols-6 gap-4">
                {visualContentSimilar.map(s => (
                  <ContentSimilarCard
                    key={s.movie_id}
                    movieId={s.movie_id}
                    navigate={navigate}
                  />
                ))}
              </div>
            </div>
          )}

          {!visualLoading && !visualPreview && (
            <p className="text-gray-500 text-center py-12">
              Carica una locandina per iniziare la ricerca visiva
            </p>
          )}
        </>
      )}
    </main>
  )
}

const FALLBACK_POSTER = 'https://via.placeholder.com/300x450/1f2937/6b7280?text=No+Poster'

function VisualRecognizedCard({ movie, navigate }) {
  const genres = movie.genres ? movie.genres.split('|') : []
  const actors = movie.actors_top5 ? movie.actors_top5.split(',').map(s => s.trim()) : []
  const overview = movie.overview_en
    ? movie.overview_en.length > 200 ? movie.overview_en.slice(0, 200) + '...' : movie.overview_en
    : null
  const confidence = Math.round((movie.similarity ?? 0) * 100)
  const methodLabel = movie.method === 'clip' ? 'CLIP' : 'phash'

  return (
    <div className="bg-gray-800 rounded-xl p-4 flex flex-col sm:flex-row gap-6 border border-gray-700">
      <img
        src={movie.poster_url || FALLBACK_POSTER}
        alt={movie.title}
        className="w-32 rounded-lg flex-shrink-0 self-start"
        onError={e => { e.target.src = FALLBACK_POSTER }}
      />
      <div className="flex flex-col gap-2 flex-1">
        <div className="flex flex-wrap gap-2 items-center">
          <h3 className="text-xl font-bold text-white">{movie.title}</h3>
          {movie.year && <span className="text-gray-400">{movie.year}</span>}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {genres.map(g => (
            <span key={g} className="bg-indigo-900/50 text-indigo-300 border border-indigo-800 px-2 py-0.5 rounded-full text-xs">
              {g}
            </span>
          ))}
        </div>

        <div className="flex gap-2 flex-wrap">
          <span className="text-xs bg-green-900/50 text-green-300 px-2 py-0.5 rounded-full">
            Riconosciuto con {confidence}% di confidenza
          </span>
          <span className="text-xs bg-gray-700 text-gray-400 px-2 py-0.5 rounded-full">
            {methodLabel}
          </span>
        </div>

        {movie.director && (
          <p className="text-gray-400 text-sm">
            <span className="text-gray-500">Regia </span>{movie.director}
          </p>
        )}
        {actors.length > 0 && (
          <p className="text-gray-400 text-sm">
            <span className="text-gray-500">Con </span>{actors.slice(0, 3).join(', ')}
          </p>
        )}
        {overview && <p className="text-gray-500 text-sm leading-relaxed">{overview}</p>}

        <button
          onClick={() => navigate(`/movies/${movie.movie_id}`)}
          className="mt-2 w-fit bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg text-sm transition-colors"
        >
          Vai alla scheda film →
        </button>
      </div>
    </div>
  )
}

const FALLBACK_SMALL = 'https://via.placeholder.com/150x225/1f2937/6b7280?text=?'

function ContentSimilarCard({ movieId, navigate }) {
  const [movie, setMovie] = useState(null)

  useEffect(() => {
    api.get(`/movies/${movieId}`).then(r => setMovie(r.data.movie)).catch(() => {})
  }, [movieId])

  if (!movie) return <div className="aspect-[2/3] bg-gray-700 animate-pulse rounded-lg" />

  const genres = movie.genres ? movie.genres.split('|').slice(0, 2) : []

  return (
    <div
      className="cursor-pointer group"
      onClick={() => navigate(`/movies/${movie.id ?? movieId}`)}
    >
      <img
        src={movie.poster_url || FALLBACK_SMALL}
        alt={movie.title}
        className="w-full aspect-[2/3] object-cover rounded-lg group-hover:opacity-80 transition-opacity"
        onError={e => { e.target.src = FALLBACK_SMALL }}
      />
      <div className="mt-1.5 px-0.5">
        <p className="text-xs text-white font-medium line-clamp-2 leading-tight">{movie.title_clean || movie.title}</p>
        <p className="text-xs text-gray-500">{movie.year}</p>
        <div className="flex flex-wrap gap-1 mt-0.5">
          {genres.map(g => (
            <span key={g} className="text-xs text-indigo-400">{g}</span>
          ))}
        </div>
      </div>
    </div>
  )
}
