export default function ExplanationBadge({ explanation }) {
  if (!explanation) return null

  let text
  if (explanation.type === 'social') {
    const avg = explanation.avg_rating_similar_users?.toFixed(1) ?? '?'
    text = `⭐ Piaciuto in media ${avg}/5 a utenti simili a te`
  } else {
    const seeds = explanation.seed_movies ?? []
    if (seeds.length > 0) {
      const titles = seeds.slice(0, 2).map(s => s.title)
      text = `Perché ti è piaciuto: ${titles.join(', ')}`
    } else if (explanation.type === 'popular') {
      text = 'Tra i più popolari'
    } else {
      text = 'Basato sulla tua cronologia'
    }
  }

  return (
    <p className="text-xs text-gray-500 mt-1 line-clamp-1" title={text}>
      {text}
    </p>
  )
}
