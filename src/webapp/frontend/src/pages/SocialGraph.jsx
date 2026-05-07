/* eslint-disable */
import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import api from '../api/client'

export default function SocialGraph() {
  const svgRef = useRef(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [stats, setStats] = useState(null)
  const [graphData, setGraphData] = useState(null)

  useEffect(() => {
    api.get('/social/graph')
      .then(res => {
        setGraphData(res.data)
        setStats(res.data.stats)
        setLoading(false)
      })
      .catch(err => {
        console.error('Social graph error:', err)
        setError('Errore nel caricamento del grafo')
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!graphData) return
    setStats({
      nodes: graphData.nodes.length,
      edges: graphData.edges.length,
      avgDeg: graphData.nodes.length
        ? (graphData.nodes.reduce((s, n) => s + n.degree, 0) / graphData.nodes.length).toFixed(1)
        : 0,
    })
    drawGraph(graphData, setSelected)
  }, [graphData])

  function drawGraph(data, onSelect) {
    const el = svgRef.current
    if (!el) return
    d3.select(el).selectAll('*').remove()

    const W = el.clientWidth || 900
    const H = el.clientHeight || 600

    const svg = d3.select(el)
      .attr('width', W)
      .attr('height', H)

    const g = svg.append('g')

    svg.call(
      d3.zoom()
        .scaleExtent([0.2, 4])
        .on('zoom', (event) => g.attr('transform', event.transform))
    )

    const nodes = data.nodes.map(d => ({ ...d }))
    const edges = data.edges.map(d => ({ ...d }))

    const maxRatings = d3.max(nodes, d => d.n_ratings) || 1
    const rScale = d3.scaleSqrt().domain([0, maxRatings]).range([5, 22])
    const colorScale = d3.scaleSequential(d3.interpolateCool).domain([0, maxRatings])

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(edges).id(d => d.id).distance(d => 120 - d.similarity * 60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collision', d3.forceCollide().radius(d => rScale(d.n_ratings) + 4))

    const link = g.append('g')
      .selectAll('line')
      .data(edges)
      .join('line')
      .attr('stroke', '#4b5563')
      .attr('stroke-opacity', 0.7)
      .attr('stroke-width', d => 1 + d.similarity * 3)

    const node = g.append('g')
      .selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => rScale(d.n_ratings))
      .attr('fill', d => colorScale(d.n_ratings))
      .attr('stroke', '#111827')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        event.stopPropagation()
        onSelect(d)
      })
      .call(
        d3.drag()
          .on('start', (event, d) => {
            if (!event.active) sim.alphaTarget(0.3).restart()
            d.fx = d.x; d.fy = d.y
          })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => {
            if (!event.active) sim.alphaTarget(0)
            d.fx = null; d.fy = null
          })
      )

    node.append('title').text(d => d.username)

    const label = g.append('g')
      .selectAll('text')
      .data(nodes.filter(d => d.n_ratings > 5))
      .join('text')
      .text(d => d.username)
      .attr('font-size', 10)
      .attr('fill', '#d1d5db')
      .attr('text-anchor', 'middle')
      .attr('dy', d => rScale(d.n_ratings) + 12)
      .style('pointer-events', 'none')

    svg.on('click', () => onSelect(null))

    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      node
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
      label
        .attr('x', d => d.x)
        .attr('y', d => d.y)
    })
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col">
      <div className="max-w-7xl mx-auto px-4 py-6 w-full flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Grafo Sociale</h1>
            <p className="text-gray-400 text-sm mt-0.5">
              Connessioni tra utenti basate sulla similarità coseno dei rating (soglia ≥ 0.3)
            </p>
          </div>
          {stats && (
            <div className="flex gap-4 text-sm text-gray-400">
              <span><span className="text-white font-semibold">{stats.nodes}</span> utenti</span>
              <span><span className="text-white font-semibold">{stats.edges}</span> connessioni</span>
              <span>grado medio <span className="text-white font-semibold">{stats.avgDeg}</span></span>
            </div>
          )}
        </div>

        <div className="flex gap-4 flex-1 min-h-0">
          <div className="flex-1 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden relative">
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center text-gray-400">
                Calcolo similarità in corso...
              </div>
            )}
            {error && (
              <div className="absolute inset-0 flex items-center justify-center text-red-400">
                Errore: {error}
              </div>
            )}
            <svg ref={svgRef} className="w-full h-full" style={{ minHeight: 520 }} />
          </div>

          {selected && (
            <div className="w-64 bg-gray-900 rounded-xl border border-gray-800 p-4 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <h2 className="font-bold text-lg text-indigo-300">{selected.username}</h2>
                <button
                  onClick={() => setSelected(null)}
                  className="text-gray-500 hover:text-white text-lg leading-none"
                >
                  ✕
                </button>
              </div>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Rating dati</span>
                  <span className="text-white font-medium">{selected.n_ratings}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Voto medio</span>
                  <span className="text-white font-medium">{selected.avg_rating} ★</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Connessioni</span>
                  <span className="text-white font-medium">{selected.degree}</span>
                </div>
              </div>
              {selected.top_movies?.length > 0 && (
                <div>
                  <p className="text-gray-400 text-xs mb-1">Film top-3</p>
                  <ul className="space-y-1">
                    {selected.top_movies.map((t, i) => (
                      <li key={i} className="text-xs text-gray-300 truncate" title={t}>
                        {i + 1}. {t}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mt-3 flex items-center gap-6 text-xs text-gray-500">
          <span>Dimensione nodo = numero di rating</span>
          <span>Spessore arco = similarità coseno</span>
          <span>Colore = attività (chiaro = più attivo)</span>
          <span>Trascina i nodi • Scroll per zoom</span>
        </div>
      </div>
    </div>
  )
}
