import { useEffect, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import api from '../api/client'
import MovieCard from '../components/MovieCard'

const GENRES = [
  '', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
  'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]

const glassInput = {
  background: 'rgba(255,255,255,0.06)',
  border: '1px solid rgba(255,255,255,0.10)',
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200"
      style={active ? {
        background: 'rgba(129,140,248,0.15)',
        border: '1px solid rgba(129,140,248,0.3)',
        color: '#a5b4fc',
      } : {
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.08)',
        color: '#94a3b8',
      }}
    >
      {children}
    </button>
  )
}

export default function MovieList() {
  const navigate = useNavigate()
  const [tab, setTab] = useState('title')

  const [movies, setMovies] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [genre, setGenre] = useState('')

  const [semanticInput, setSemanticInput] = useState('')
  const [semanticResults, setSemanticResults] = useState([])
  const [semanticQuery, setSemanticQuery] = useState('')
  const [semanticLoading, setSemanticLoading] = useState(false)
  const [semanticError, setSemanticError] = useState('')

  const [visualFile, setVisualFile] = useState(null)
  const [visualPreview, setVisualPreview] = useState(null)
  const [visualRecognized, setVisualRecognized] = useState(null)
  const [visualSimilar, setVisualSimilar] = useState([])
  const [visualContentSimilar, setVisualContentSimilar] = useState([])
  const [visualLoading, setVisualLoading] = useState(false)
  const [visualError, setVisualError] = useState('')
  const [visualNotFound, setVisualNotFound] = useState(false)

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
    <main className="pt-5 sm:pt-6 pb-12 px-3 sm:px-4 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-xl sm:text-2xl font-bold text-slate-100 mb-4">Catalogo Film</h1>

        {/* Tab switcher */}
        <div className="flex gap-2 mb-4 flex-wrap">
          <TabButton active={tab === 'title'} onClick={() => setTab('title')}>🔍 Titolo</TabButton>
          <TabButton active={tab === 'semantic'} onClick={() => setTab('semantic')}>✨ Concetto</TabButton>
          <TabButton active={tab === 'visual'} onClick={() => setTab('visual')}>🖼️ Foto locandina</TabButton>
        </div>
      </div>

      {tab === 'title' && (
        <>
          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <form onSubmit={handleSearch} className="flex flex-col min-[380px]:flex-row gap-2 flex-1">
              <input
                type="text"
                value={searchInput}
                onChange={e => setSearchInput(e.target.value)}
                placeholder="Cerca per titolo..."
                className="flex-1 rounded-xl px-4 py-2.5 text-slate-100 text-sm placeholder:text-slate-600 outline-none"
                style={glassInput}
              />
              <button
                type="submit"
                className="text-white px-4 py-2 rounded-xl text-sm transition-all"
                style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
              >
                Cerca
              </button>
            </form>
            <select
              value={genre}
              onChange={e => handleGenre(e.target.value)}
              className="rounded-xl px-3 py-2.5 text-sm text-slate-300 outline-none"
              style={glassInput}
            >
              <option value="">Tutti i generi</option>
              {GENRES.filter(Boolean).map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>

          {loading ? (
            <div className="flex justify-center py-16 text-slate-400 text-2xl animate-spin">⟳</div>
          ) : (
            <>
              <p className="text-slate-500 text-sm mb-4">{total} film trovati</p>
              <div className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3 mb-8">
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
                <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4">
                  <button
                    onClick={() => setPage(p => Math.max(1, p - 1))}
                    disabled={page === 1}
                    className="px-4 py-2 rounded-xl text-sm text-slate-300 disabled:opacity-40 transition-all"
                    style={glassInput}
                  >
                    ← Precedente
                  </button>
                  <span className="text-slate-400 text-sm">Pagina {page} di {totalPages}</span>
                  <button
                    onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                    className="px-4 py-2 rounded-xl text-sm text-slate-300 disabled:opacity-40 transition-all"
                    style={glassInput}
                  >
                    Successiva →
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {tab === 'semantic' && (
        <>
          <form onSubmit={handleSemanticSearch} className="flex flex-col min-[420px]:flex-row gap-2 mb-2">
            <input
              type="text"
              value={semanticInput}
              onChange={e => setSemanticInput(e.target.value)}
              placeholder="es. film con maghi anni 70, thriller psicologico, storia d'amore a Parigi..."
              className="flex-1 rounded-xl px-4 py-2.5 text-slate-100 text-sm placeholder:text-slate-600 outline-none"
              style={glassInput}
            />
            <button
              type="submit"
              disabled={semanticLoading || semanticInput.trim().length < 3}
              className="text-white px-4 py-2 rounded-xl text-sm disabled:opacity-40 transition-all whitespace-nowrap"
              style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
            >
              {semanticLoading ? '⟳' : '✨ Cerca'}
            </button>
          </form>
          <p className="text-xs text-slate-500 mb-6">
            Cerca per trama, ambientazione, periodo storico, mood o qualsiasi concetto — non solo titolo
          </p>

          {semanticError && <p className="text-red-400 text-sm mb-4">{semanticError}</p>}

          {semanticResults.length > 0 && (
            <>
              <p className="text-slate-500 text-sm mb-4">
                {semanticResults.length} risultati per "{semanticQuery}"
              </p>
              <div className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3 sm:gap-4">
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
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{
                          background: 'rgba(99,102,241,0.15)',
                          color: '#a5b4fc',
                          border: '1px solid rgba(99,102,241,0.25)',
                        }}
                      >
                        Pertinenza: {Math.round(m.similarity_score * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {!semanticLoading && semanticQuery && semanticResults.length === 0 && !semanticError && (
            <p className="text-slate-500 text-center py-12">Nessun risultato trovato per "{semanticQuery}"</p>
          )}
        </>
      )}

      {tab === 'visual' && (
        <>
          <div className="mb-6">
            <label className="flex items-center gap-3 cursor-pointer w-fit">
              <span
                className="text-white px-4 py-2 rounded-xl text-sm font-medium transition-all"
                style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
              >
                📁 Carica foto locandina
              </span>
              <input type="file" accept="image/*" className="hidden" onChange={handleVisualFileChange} />
            </label>
            <p className="text-xs text-slate-500 mt-2">
              Carica una foto di una locandina per trovare il film o film con poster simili
            </p>
          </div>

          {visualPreview && (
            <div className="flex flex-col min-[420px]:flex-row items-start gap-4 sm:gap-6 mb-6">
              <img
                src={visualPreview}
                alt="preview"
                className="w-24 h-36 object-cover rounded-xl"
                style={{ border: '1px solid rgba(255,255,255,0.15)' }}
              />
              <div className="flex flex-col gap-2">
                <p className="text-slate-400 text-sm">{visualFile?.name}</p>
                <button
                  onClick={handleVisualSearch}
                  disabled={visualLoading}
                  className="text-white px-4 py-2 rounded-xl text-sm disabled:opacity-40 transition-all w-fit"
                  style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
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

          {visualRecognized && (
            <div className="mb-8">
              <h2 className="text-base font-semibold text-slate-200 mb-4">Film riconosciuto</h2>
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

          {visualSimilar.length > 0 && (
            <div className="mb-8">
              <h2 className="text-base font-semibold text-slate-200 mb-4">Film con poster simili</h2>
              <div className="flex gap-3 overflow-x-auto pb-2 -mx-3 px-3 sm:mx-0 sm:px-0">
                {visualSimilar.slice(0, 5).map(m => (
                  <div key={m.movie_id} className="flex-shrink-0 w-36 sm:w-40 flex flex-col">
                    <MovieCard
                      movie={{ ...m, id: m.movie_id }}
                      userRating={userRatings[m.movie_id]}
                      isFavorite={favorites.has(m.movie_id)}
                      onRate={handleRate}
                      onFavorite={handleFavorite}
                    />
                    <div className="mt-1 px-1">
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{
                          background: 'rgba(139,92,246,0.15)',
                          color: '#c4b5fd',
                          border: '1px solid rgba(139,92,246,0.25)',
                        }}
                      >
                        Visiva: {Math.round(m.visual_similarity * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {visualContentSimilar.length > 0 && visualRecognized && (
            <div className="mb-8">
              <h2 className="text-base font-semibold text-slate-200 mb-4">
                Ti potrebbe piacere anche —{' '}
                <span className="text-slate-400 font-normal">basato sulla trama di {visualRecognized.title}</span>
              </h2>
              <div className="grid grid-cols-[repeat(auto-fit,minmax(110px,1fr))] gap-3">
                {visualContentSimilar.map(s => (
                  <ContentSimilarCard key={s.movie_id} movieId={s.movie_id} navigate={navigate} />
                ))}
              </div>
            </div>
          )}

          {!visualLoading && !visualPreview && (
            <p className="text-slate-500 text-center py-12">
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
    <div
      className="rounded-2xl p-4 flex flex-col sm:flex-row gap-6"
      style={{
        background: 'rgba(255,255,255,0.05)',
        border: '1px solid rgba(255,255,255,0.10)',
        backdropFilter: 'blur(12px)',
      }}
    >
      <img
        src={movie.poster_url || FALLBACK_POSTER}
        alt={movie.title}
        className="w-32 rounded-xl flex-shrink-0 self-start"
        onError={e => { e.target.src = FALLBACK_POSTER }}
      />
      <div className="flex flex-col gap-2 flex-1">
        <div className="flex flex-wrap gap-2 items-center">
          <h3 className="text-xl font-bold text-slate-100">{movie.title}</h3>
          {movie.year && <span className="text-slate-400">{movie.year}</span>}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {genres.map(g => (
            <span
              key={g}
              className="px-2 py-0.5 rounded-full text-xs"
              style={{
                background: 'rgba(129,140,248,0.12)',
                border: '1px solid rgba(129,140,248,0.25)',
                color: '#a5b4fc',
              }}
            >
              {g}
            </span>
          ))}
        </div>
        <div className="flex gap-2 flex-wrap">
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(34,197,94,0.15)', color: '#86efac', border: '1px solid rgba(34,197,94,0.25)' }}>
            Riconosciuto con {confidence}% di confidenza
          </span>
          <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.10)' }}>
            {methodLabel}
          </span>
        </div>
        {movie.director && (
          <p className="text-slate-400 text-sm"><span className="text-slate-500">Regia </span>{movie.director}</p>
        )}
        {actors.length > 0 && (
          <p className="text-slate-400 text-sm"><span className="text-slate-500">Con </span>{actors.slice(0, 3).join(', ')}</p>
        )}
        {overview && <p className="text-slate-500 text-sm leading-relaxed">{overview}</p>}
        <button
          onClick={() => navigate(`/movies/${movie.movie_id}`)}
          className="mt-2 w-fit text-white px-4 py-2 rounded-xl text-sm transition-all"
          style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
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

  if (!movie) return <div className="aspect-[2/3] animate-pulse rounded-xl" style={{ background: 'rgba(255,255,255,0.06)' }} />

  const genres = movie.genres ? movie.genres.split('|').slice(0, 2) : []

  return (
    <div className="cursor-pointer group" onClick={() => navigate(`/movies/${movie.id ?? movieId}`)}>
      <img
        src={movie.poster_url || FALLBACK_SMALL}
        alt={movie.title}
        className="w-full aspect-[2/3] object-cover rounded-xl group-hover:opacity-80 transition-opacity"
        onError={e => { e.target.src = FALLBACK_SMALL }}
      />
      <div className="mt-1.5 px-0.5">
        <p className="text-xs text-slate-200 font-medium line-clamp-2 leading-tight">{movie.title_clean || movie.title}</p>
        <p className="text-xs text-slate-500">{movie.year}</p>
        <div className="flex flex-wrap gap-1 mt-0.5">
          {genres.map(g => <span key={g} className="text-xs text-indigo-400">{g}</span>)}
        </div>
      </div>
    </div>
  )
}
