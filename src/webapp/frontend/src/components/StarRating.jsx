import { useState } from 'react'

export default function StarRating({ value = 0, onChange, readonly = false, size = 'md' }) {
  const [hover, setHover] = useState(null)

  const sizeClass = size === 'sm' ? 'text-base' : 'text-xl'
  const display = hover ?? value

  return (
    <div className={`flex gap-0.5 ${sizeClass}`}>
      {[1, 2, 3, 4, 5].map(star => {
        const full = display >= star
        const half = !full && display >= star - 0.5
        return (
          <span
            key={star}
            className={`cursor-${readonly ? 'default' : 'pointer'} relative inline-block`}
            onMouseLeave={readonly ? undefined : () => setHover(null)}
          >
            {/* left half */}
            {!readonly && (
              <span
                className="absolute left-0 top-0 w-1/2 h-full z-10"
                onMouseEnter={() => setHover(star - 0.5)}
                onClick={() => onChange?.(star - 0.5)}
              />
            )}
            {/* right half */}
            {!readonly && (
              <span
                className="absolute right-0 top-0 w-1/2 h-full z-10"
                onMouseEnter={() => setHover(star)}
                onClick={() => onChange?.(star)}
              />
            )}
            <span className={full ? 'text-yellow-400' : half ? 'text-yellow-400' : 'text-gray-600'}>
              {full ? '★' : half ? '⯨' : '☆'}
            </span>
          </span>
        )
      })}
    </div>
  )
}
