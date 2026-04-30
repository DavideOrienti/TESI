import { useNavigate } from 'react-router-dom'
import StarRating from './StarRating'
import ExplanationBadge from './ExplanationBadge'

const FALLBACK = 'https://via.placeholder.com/300x450/1f2937/6b7280?text=No+Poster'

export default function MovieCard({ movie, userRating, isFavorite, onRate, onFavorite, explanation, showRating = true, llmExplanation = null, onExplain = null }) {
  const navigate = useNavigate()
  const genres = movie.genres ? movie.genres.split('|').slice(0, 2) : []

  function handleCardClick(e) {
    if (e.target.closest('[data-no-nav]')) return
    navigate(`/movies/${movie.id ?? movie.movie_id}`)
  }

  return (
    <div
      className="bg-gray-800 rounded-lg overflow-hidden cursor-pointer transition-transform duration-200 hover:scale-105 hover:shadow-xl flex flex-col"
      onClick={handleCardClick}
    >
      {/* Poster */}
      <div className="relative">
        <img
          src={movie.poster_url || FALLBACK}
          alt={movie.title}
          className="w-full aspect-[2/3] object-cover"
          onError={e => { e.target.src = FALLBACK }}
          loading="lazy"
        />
        {/* Favorite button */}
        {onFavorite && (
          <button
            data-no-nav
            onClick={e => { e.stopPropagation(); onFavorite(movie.id ?? movie.movie_id) }}
            className="absolute top-2 right-2 text-xl drop-shadow"
            title={isFavorite ? 'Rimuovi dai preferiti' : 'Aggiungi ai preferiti'}
          >
            {isFavorite ? '❤️' : '🤍'}
          </button>
        )}
      </div>

      {/* Info */}
      <div className="p-3 flex flex-col gap-1 flex-1">
        <h3 className="text-sm font-semibold text-white line-clamp-2 leading-tight">
          {movie.title_clean || movie.title}
        </h3>
        <p className="text-xs text-gray-400">{movie.year}</p>
        {genres.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {genres.map(g => (
              <span key={g} className="text-xs bg-gray-700 text-gray-300 px-1.5 py-0.5 rounded">
                {g}
              </span>
            ))}
          </div>
        )}

        {/* Stars */}
        {showRating && (
          <div data-no-nav className="mt-1">
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
            style={{ fontSize: '0.7rem', color: '#6366f1', background: 'none', border: 'none', cursor: 'pointer', padding: '2px 0', marginTop: '4px' }}
          >
            ✨ Spiega perché
          </button>
        )}
        {llmExplanation?.loading && (
          <p className="text-xs text-gray-500 italic animate-pulse mt-1">
            Generazione spiegazione personalizzata...
          </p>
        )}
        {llmExplanation?.text && (
          <span style={{ fontSize: '0.75rem', color: '#9ca3af', fontStyle: 'italic', marginTop: '4px', lineHeight: '1.4', display: 'block' }}>
            {llmExplanation.text}
          </span>
        )}
      </div>
    </div>
  )
}
