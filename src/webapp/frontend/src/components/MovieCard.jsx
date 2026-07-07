import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import StarRating from './StarRating'
import ExplanationBadge from './ExplanationBadge'

const FALLBACK = 'https://via.placeholder.com/300x450/1f2937/6b7280?text=No+Poster'

export default function MovieCard({ movie, userRating, isFavorite, onRate, onFavorite, explanation, showRating = true, llmExplanation = null, onExplain = null }) {
  const navigate = useNavigate()
  const genres = movie.genres ? movie.genres.split('|').slice(0, 2) : []
  const [hovered, setHovered] = useState(false)

  function handleCardClick(e) {
    if (e.target.closest('[data-no-nav]')) return
    navigate(`/movies/${movie.id ?? movie.movie_id}`)
  }

  return (
    <div
      className="group relative min-w-0 rounded-xl sm:rounded-2xl overflow-hidden cursor-pointer transition-all duration-300 sm:hover:scale-[1.02] sm:hover:-translate-y-1"
      style={{
        background: 'rgba(255,255,255,0.05)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255,255,255,0.10)',
        boxShadow: hovered
          ? '0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(129,140,248,0.2)'
          : '0 4px 24px rgba(0,0,0,0.3)',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={handleCardClick}
    >
      {/* Poster */}
      <div className="relative w-full overflow-hidden" style={{ aspectRatio: '2/3' }}>
        <img
          src={movie.poster_url || FALLBACK}
          alt={movie.title}
          className="w-full h-full object-cover object-center absolute inset-0"
          onError={e => { e.target.src = FALLBACK }}
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-black/10 to-transparent" />

        {/* Anno badge */}
        {movie.year && (
          <span className="absolute bottom-2 left-2 text-xs text-white/80 bg-black/50 backdrop-blur-sm rounded-full px-2 py-0.5">
            {movie.year}
          </span>
        )}

        {/* Cuore preferiti */}
        {onFavorite && (
          <button
            data-no-nav
            onClick={e => { e.stopPropagation(); onFavorite(movie.id ?? movie.movie_id) }}
            className="absolute top-2 right-2 bg-black/40 backdrop-blur-sm rounded-full p-1.5 transition-all duration-200 opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
            title={isFavorite ? 'Rimuovi dai preferiti' : 'Aggiungi ai preferiti'}
          >
            <span className={isFavorite ? 'text-rose-400' : 'text-white/60'}>
              {isFavorite ? '❤️' : '🤍'}
            </span>
          </button>
        )}
      </div>

      {/* Body */}
      <div className="min-w-0 p-2.5 sm:p-3">
        <h3 className="font-medium text-sm text-slate-100 leading-tight line-clamp-2 break-words mb-1.5">
          {movie.title_clean || movie.title}
        </h3>

        {genres.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-2">
            {genres.map(g => (
              <span
                key={g}
                className="max-w-full truncate text-[10px] px-1.5 sm:px-2 py-0.5 rounded-full border"
                style={{
                  background: 'rgba(129,140,248,0.12)',
                  borderColor: 'rgba(129,140,248,0.25)',
                  color: '#a5b4fc',
                }}
              >
                {g}
              </span>
            ))}
          </div>
        )}

        {showRating && (
          <div data-no-nav className="mt-1 overflow-hidden">
            <StarRating
              value={userRating ?? 0}
              onChange={onRate ? (v) => onRate(movie.id ?? movie.movie_id, v) : undefined}
              readonly={!onRate}
              size="sm"
            />
          </div>
        )}

        {explanation && <ExplanationBadge explanation={explanation} />}

        {onExplain && !llmExplanation && (
          <button
            data-no-nav
            onClick={e => { e.stopPropagation(); onExplain(movie.id ?? movie.movie_id) }}
            className="text-[10px] text-indigo-400 hover:text-indigo-300 mt-1 transition-colors duration-150"
            style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
          >
            ✨ Spiega perché
          </button>
        )}
        {llmExplanation === 'loading' && (
          <p className="text-[11px] text-slate-500 italic animate-pulse mt-1">
            Generazione spiegazione personalizzata...
          </p>
        )}
        {llmExplanation && llmExplanation !== 'loading' && (
          <p className="text-[11px] italic text-slate-400 mt-1" style={{ lineHeight: 1.4 }}>
            {llmExplanation}
          </p>
        )}
      </div>
    </div>
  )
}
