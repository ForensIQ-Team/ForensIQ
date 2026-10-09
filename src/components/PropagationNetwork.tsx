import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import {
  Globe,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  ExternalLink,
  Copy,
  Check,
  X,
  Calendar,
  User,
  MapPin,
  Clock,
  Layers,
  Search as SearchIcon,
  ArrowUpRight,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import {
  PropagationResult,
  matchTypeLabel,
} from '../services/propagationService';

// ================================================================
// PropagationNetwork — dark violet, collapsible, no-overflow view.
//
// Design:
//   • Hub card (left) — investigated media + source count
//   • Collapsible platform cards (right) — grid of fixed-width
//     columns. Each card expands to reveal its sources inside a
//     max-height scroll area, so the column NEVER grows vertically
//     unbounded.
//   • Connectors are drawn on an SVG layer behind the cards.
//   • Only "Visual Match" terminology is surfaced (Exact is hidden).
//
// Data contract unchanged:
//   props.results              : PropagationResult[]
//   props.searchedImagePreview : string | null
// ================================================================

interface PropagationNetworkProps {
  results: PropagationResult[];
  searchedImagePreview: string | null;
}

type PlatformCategory =
  | 'Social Media'
  | 'News & Media'
  | 'Video Platforms'
  | 'Image Platforms'
  | 'Forums & Communities'
  | 'E-Commerce'
  | 'Blogs'
  | 'General Web';

const SOCIAL_KEYWORDS = ['reddit', 'instagram', 'facebook', 'twitter', 'x.com', 'tiktok', 'pinterest', 'tumblr', 'linkedin', 'snapchat', 'weibo', 'vk.com', 'threads'];
const NEWS_KEYWORDS = ['bbc', 'cnn', 'nytimes', 'reuters', 'theguardian', 'news', 'times', 'post', 'herald', 'tribune', 'daily', 'press', 'media', 'journal', 'gazette', 'magazine', 'espn', 'huffpost', 'buzzfeed', 'gizmodo', 'techcrunch'];
const VIDEO_KEYWORDS = ['youtube', 'vimeo', 'dailymotion', 'twitch', 'rumble', 'odysee', 'bitchute'];
const IMAGE_KEYWORDS = ['flickr', 'imgur', 'deviantart', 'artstation', 'behance', '500px', 'shutterstock', 'gettyimages', 'unsplash', 'pexels', 'alamy', 'dreamstime', 'istockphoto'];
const FORUM_KEYWORDS = ['forum', 'quora', 'stackexchange', 'stackoverflow', 'board', 'discuss', 'community'];
const ECOMMERCE_KEYWORDS = ['amazon', 'ebay', 'etsy', 'shopify', 'store', 'shop', 'market', 'aliexpress', 'walmart', 'target', 'bestbuy', 'wayfair'];
const BLOG_KEYWORDS = ['blog', 'wordpress', 'blogspot', 'medium', 'substack', 'ghost'];

function classifyDomain(domain: string | null): PlatformCategory {
  if (!domain) return 'General Web';
  const d = domain.toLowerCase();
  if (SOCIAL_KEYWORDS.some(k => d.includes(k))) return 'Social Media';
  if (VIDEO_KEYWORDS.some(k => d.includes(k))) return 'Video Platforms';
  if (IMAGE_KEYWORDS.some(k => d.includes(k))) return 'Image Platforms';
  if (NEWS_KEYWORDS.some(k => d.includes(k))) return 'News & Media';
  if (ECOMMERCE_KEYWORDS.some(k => d.includes(k))) return 'E-Commerce';
  if (BLOG_KEYWORDS.some(k => d.includes(k))) return 'Blogs';
  if (FORUM_KEYWORDS.some(k => d.includes(k))) return 'Forums & Communities';
  return 'General Web';
}

// Category palette tuned for the dark violet canvas.
const CATEGORY_CONFIG: Record<
  PlatformCategory,
  { accent: string; soft: string; border: string; text: string; dot: string }
> = {
  'Social Media':         { accent: '#8b5cf6', soft: 'rgba(139,92,246,0.14)', border: 'rgba(139,92,246,0.38)', text: '#c4b5fd', dot: '#8b5cf6' },
  'News & Media':         { accent: '#f43f5e', soft: 'rgba(244,63,94,0.14)',  border: 'rgba(244,63,94,0.38)',  text: '#fda4af', dot: '#f43f5e' },
  'Video Platforms':      { accent: '#fb923c', soft: 'rgba(251,146,60,0.14)', border: 'rgba(251,146,60,0.38)', text: '#fdba74', dot: '#fb923c' },
  'Image Platforms':      { accent: '#22d3ee', soft: 'rgba(34,211,238,0.14)', border: 'rgba(34,211,238,0.38)', text: '#67e8f9', dot: '#22d3ee' },
  'Forums & Communities': { accent: '#a78bfa', soft: 'rgba(167,139,250,0.14)',border: 'rgba(167,139,250,0.38)',text: '#c4b5fd', dot: '#a78bfa' },
  'E-Commerce':           { accent: '#34d399', soft: 'rgba(52,211,153,0.14)', border: 'rgba(52,211,153,0.38)', text: '#6ee7b7', dot: '#34d399' },
  'Blogs':                { accent: '#fbbf24', soft: 'rgba(251,191,36,0.14)', border: 'rgba(251,191,36,0.38)', text: '#fcd34d', dot: '#fbbf24' },
  'General Web':          { accent: '#a1a1aa', soft: 'rgba(161,161,170,0.14)',border: 'rgba(161,161,170,0.34)',text: '#d4d4d8', dot: '#a1a1aa' },
};

function extractPoster(r: PropagationResult): string {
  if (r.snippet) {
    const handle = r.snippet.match(/@[A-Za-z0-9_.]{2,28}/);
    if (handle) return handle[0];
    const byLine = r.snippet.match(/\b(?:by|author|posted by)\s+([A-Z][a-zA-Z\s]{2,24})/i);
    if (byLine) return byLine[1].trim();
  }
  return 'Not available';
}

// ================================================================
// Component
// ================================================================
export const PropagationNetwork: React.FC<PropagationNetworkProps> = ({
  results,
  searchedImagePreview,
}) => {
  const [selectedResult, setSelectedResult] = useState<PropagationResult | null>(null);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Which category cards are expanded — default: top 2 by source count
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const canvasRef = useRef<HTMLDivElement>(null);
  const hubRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());
  const [connectors, setConnectors] = useState<
    Array<{ category: PlatformCategory; d: string; accent: string }>
  >([]);

  // ── Group results ─────────────────────────────────────────────
  const groups = useMemo(() => {
    const map = new Map<PlatformCategory, PropagationResult[]>();
    results.forEach(r => {
      const cat = classifyDomain(r.domain);
      if (!map.has(cat)) map.set(cat, []);
      map.get(cat)!.push(r);
    });
    return Array.from(map.entries())
      .sort((a, b) => b[1].length - a[1].length)
      .map(([category, items]) => ({ category, items }));
  }, [results]);

  // Auto-expand top 2 groups when data first arrives / changes
  useEffect(() => {
    const top = groups.slice(0, 2).map(g => g.category);
    setExpanded(new Set(top));
  }, [groups]);

  const toggleExpand = (cat: string) =>
    setExpanded(prev => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });

  // ── Timeline ──────────────────────────────────────────────────
  const timelineItems = useMemo(() => {
    return results
      .filter(r => r.date && r.date.trim())
      .sort((a, b) => (a.date! > b.date! ? 1 : -1))
      .slice(0, 12);
  }, [results]);

  // ── Draw connectors ───────────────────────────────────────────
  const drawConnectors = useCallback(() => {
    const canvas = canvasRef.current;
    const hub = hubRef.current;
    if (!canvas || !hub) return;
    const canvasRect = canvas.getBoundingClientRect();
    const hubRect = hub.getBoundingClientRect();

    const hubX = hubRect.right - canvasRect.left;
    const hubY = hubRect.top + hubRect.height / 2 - canvasRect.top;

    const next: Array<{ category: PlatformCategory; d: string; accent: string }> = [];

    cardRefs.current.forEach((el, category) => {
      if (!el) return;
      const r = el.getBoundingClientRect();
      const targetX = r.left - canvasRect.left;
      const targetY = r.top + Math.min(40, r.height / 2) - canvasRect.top;

      const dx = (targetX - hubX) / 2;
      const d = `M ${hubX} ${hubY} C ${hubX + dx} ${hubY}, ${targetX - dx} ${targetY}, ${targetX} ${targetY}`;

      next.push({
        category: category as PlatformCategory,
        d,
        accent: CATEGORY_CONFIG[category as PlatformCategory].accent,
      });
    });

    setConnectors(next);
  }, []);

  useEffect(() => {
    const t = setTimeout(drawConnectors, 40);
    const t2 = setTimeout(drawConnectors, 260); // after layout settles
    const handleResize = () => drawConnectors();
    window.addEventListener('resize', handleResize);
    return () => {
      clearTimeout(t);
      clearTimeout(t2);
      window.removeEventListener('resize', handleResize);
    };
  }, [drawConnectors, groups, zoom, expanded]);

  const copyUrl = (url: string) => {
    navigator.clipboard.writeText(url);
    setCopiedUrl(true);
    setTimeout(() => setCopiedUrl(false), 2000);
  };

  if (results.length === 0) {
    return (
      <div className="rounded-xl p-10 text-center border border-white/10 bg-white/5 backdrop-blur-sm space-y-2">
        <Globe className="w-8 h-8 text-violet-300/70 mx-auto" />
        <h4 className="font-bold text-white text-sm">Propagation Network Unavailable</h4>
        <p className="text-xs text-violet-100/60 max-w-xs mx-auto">
          No public visual matches were discovered for this search.
        </p>
      </div>
    );
  }

  // ----------------------------------------------------------------
  // RENDER
  // ----------------------------------------------------------------
  return (
    <div className="rounded-xl overflow-hidden border border-white/10 bg-[#1a1530]/70 backdrop-blur-md shadow-2xl shadow-violet-900/30">

      {/* Toolbar */}
      <div className="px-5 pt-4 pb-3 flex flex-wrap items-center justify-between gap-3 border-b border-white/10">
        <div>
          <h3 className="font-bold text-white text-sm uppercase tracking-wider flex items-center gap-2">
            <Globe className="w-4 h-4 text-violet-300" />
            Propagation Network
          </h3>
          <p className="text-[11px] text-violet-100/60 mt-0.5">
            {results.length} visual match{results.length !== 1 ? 'es' : ''} across {groups.length} platform group{groups.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setZoom(z => Math.min(1.6, +(z + 0.1).toFixed(2)))}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 cursor-pointer"
            title="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5 text-violet-100" />
          </button>
          <button
            onClick={() => setZoom(z => Math.max(0.6, +(z - 0.1).toFixed(2)))}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 cursor-pointer"
            title="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5 text-violet-100" />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-1.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 cursor-pointer"
            title="Reset zoom"
          >
            <RotateCcw className="w-3.5 h-3.5 text-violet-100" />
          </button>
          <span className="text-[10px] font-mono text-violet-100/50 ml-1">
            {Math.round(zoom * 100)}%
          </span>
        </div>
      </div>

      {/* Body: canvas | timeline */}
      <div className="flex" style={{ minHeight: 620 }}>

        {/* ─── Main canvas ─────────────────────────────────────── */}
        <div
          ref={canvasRef}
          className="relative flex-1 overflow-hidden"
          style={{
            background:
              'radial-gradient(circle at 15% 10%, rgba(161,115,255,0.20) 0%, rgba(60,42,110,0.30) 40%, rgba(18,14,36,0.95) 100%), linear-gradient(180deg, #1c1533 0%, #120d24 100%)',
          }}
        >
          {/* Violet dot grid */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage:
                'radial-gradient(circle, rgba(196,181,253,0.22) 1px, transparent 1px)',
              backgroundSize: '26px 26px',
              opacity: 0.7,
            }}
          />
          {/* Soft pink glow top-right */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background:
                'radial-gradient(circle at 85% 15%, rgba(225,160,214,0.18) 0%, transparent 45%)',
            }}
          />

          {/* SVG connectors */}
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none"
            style={{ zIndex: 1 }}
          >
            <defs>
              {Object.entries(CATEGORY_CONFIG).map(([k, cfg]) => (
                <marker
                  key={k}
                  id={`arrow-${k.replace(/\s|&/g, '_')}`}
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={cfg.dot} />
                </marker>
              ))}
            </defs>

            {connectors.map((c, i) => (
              <g key={`${c.category}-${i}`}>
                <path
                  d={c.d}
                  fill="none"
                  stroke={c.accent}
                  strokeWidth={8}
                  strokeOpacity={0.12}
                  strokeLinecap="round"
                />
                <path
                  d={c.d}
                  fill="none"
                  stroke={c.accent}
                  strokeWidth={1.8}
                  strokeOpacity={0.85}
                  strokeLinecap="round"
                  markerEnd={`url(#arrow-${c.category.replace(/\s|&/g, '_')})`}
                />
              </g>
            ))}
          </svg>

          {/* Cards layer */}
          <div
            className="relative p-6"
            style={{
              zIndex: 2,
              transform: `scale(${zoom})`,
              transformOrigin: 'top left',
              transition: 'transform 140ms ease-out',
            }}
          >
            <div className="flex flex-col lg:flex-row gap-8 items-start">

              {/* ─── HUB ─── */}
              <div
                ref={hubRef}
                className="shrink-0 w-full lg:w-72 rounded-2xl border border-violet-400/25 bg-gradient-to-b from-violet-500/10 to-fuchsia-500/5 backdrop-blur-md shadow-xl shadow-violet-900/30 overflow-hidden"
              >
                <div className="px-4 py-3 border-b border-white/10 bg-white/5">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-violet-300" />
                    <span className="text-[10px] font-extrabold text-violet-200 uppercase tracking-widest">
                      Investigated Media
                    </span>
                  </div>
                </div>

                <div className="p-4 space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-16 h-16 rounded-xl overflow-hidden border-2 border-violet-400/60 shadow-lg shadow-violet-900/40 shrink-0 bg-white/5">
                      {searchedImagePreview ? (
                        <img
                          src={searchedImagePreview}
                          alt="Investigated"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Globe className="w-7 h-7 text-violet-300/60" />
                        </div>
                      )}
                    </div>
                    <div className="min-w-0">
                      <p className="text-[10px] font-bold text-violet-300 uppercase tracking-wider">
                        Search Root
                      </p>
                      <p className="text-xs font-semibold text-white mt-0.5">
                        Public web propagation
                      </p>
                      <p className="text-[10px] text-violet-100/60 mt-0.5">
                        {results.length} visual match{results.length !== 1 ? 'es' : ''} · {groups.length} platform{groups.length !== 1 ? 's' : ''}
                      </p>
                    </div>
                  </div>

                  {/* Visual matches stat */}
                  <div className="rounded-lg border border-amber-400/25 bg-amber-400/5 px-3 py-2.5">
                    <p className="text-[9px] font-bold text-amber-300/80 uppercase tracking-wider">
                      Visual Matches
                    </p>
                    <p className="text-xl font-extrabold text-amber-300 leading-tight mt-0.5">
                      {results.length}
                    </p>
                  </div>

                  {/* Category chips */}
                  <div className="space-y-1.5 pt-1 border-t border-white/10">
                    <p className="text-[9px] font-bold text-violet-200/70 uppercase tracking-wider pt-2">
                      Groups
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {groups.map(g => {
                        const cfg = CATEGORY_CONFIG[g.category];
                        return (
                          <span
                            key={g.category}
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-bold border backdrop-blur-sm"
                            style={{
                              background: cfg.soft,
                              borderColor: cfg.border,
                              color: cfg.text,
                            }}
                          >
                            <span className="w-1.5 h-1.5 rounded-full" style={{ background: cfg.dot }} />
                            {g.category} · {g.items.length}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>

              {/* ─── CATEGORY GRID (collapsible cards) ─── */}
              <div
                className="flex-1 grid gap-5 w-full items-start"
                style={{
                  gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
                }}
              >
                {groups.map(g => {
                  const cfg = CATEGORY_CONFIG[g.category];
                  const isOpen = expanded.has(g.category);
                  const preview = g.items.slice(0, 3);

                  return (
                    <div
                      key={g.category}
                      ref={el => {
                        if (el) cardRefs.current.set(g.category, el);
                        else cardRefs.current.delete(g.category);
                      }}
                      className="rounded-2xl border backdrop-blur-md shadow-lg shadow-black/30 overflow-hidden transition-all"
                      style={{
                        borderColor: cfg.border,
                        background: 'rgba(255,255,255,0.04)',
                      }}
                    >
                      {/* Header — clickable */}
                      <button
                        onClick={() => toggleExpand(g.category)}
                        className="w-full px-4 py-3 flex items-center justify-between gap-2 cursor-pointer hover:bg-white/5 transition-colors"
                        style={{
                          background: cfg.soft,
                          borderBottom: isOpen ? `1px solid ${cfg.border}` : 'none',
                        }}
                      >
                        <div className="flex items-center gap-2 min-w-0">
                          {isOpen
                            ? <ChevronDown className="w-3.5 h-3.5 shrink-0" style={{ color: cfg.text }} />
                            : <ChevronRight className="w-3.5 h-3.5 shrink-0" style={{ color: cfg.text }} />
                          }
                          <span
                            className="w-2 h-2 rounded-full shrink-0"
                            style={{ background: cfg.dot }}
                          />
                          <span
                            className="text-[11px] font-extrabold uppercase tracking-widest truncate"
                            style={{ color: cfg.text }}
                          >
                            {g.category}
                          </span>
                        </div>
                        <span
                          className="text-[10px] font-bold px-1.5 py-0.5 rounded shrink-0 border"
                          style={{
                            background: 'rgba(0,0,0,0.25)',
                            color: cfg.text,
                            borderColor: cfg.border,
                          }}
                        >
                          {g.items.length}
                        </span>
                      </button>

                      {/* Collapsed → preview chips */}
                      {!isOpen && (
                        <div className="px-3 py-3 space-y-1.5">
                          {preview.map((r, i) => {
                            const domain = (r.source || r.domain || 'Web').replace(/^www\./, '');
                            return (
                              <div
                                key={r.url ?? `${g.category}-${i}`}
                                className="text-[10px] text-violet-100/60 truncate flex items-center gap-1.5"
                              >
                                <span
                                  className="w-1 h-1 rounded-full shrink-0"
                                  style={{ background: cfg.dot, opacity: 0.7 }}
                                />
                                <span className="truncate">{domain}</span>
                              </div>
                            );
                          })}
                          {g.items.length > preview.length && (
                            <p className="text-[9px] text-violet-200/40 italic pl-2.5 pt-0.5">
                              + {g.items.length - preview.length} more — click to expand
                            </p>
                          )}
                        </div>
                      )}

                      {/* Expanded → full scrollable list */}
                      {isOpen && (
                        <div
                          className="p-3 space-y-2 overflow-y-auto propagation-scroll"
                          style={{ maxHeight: 420 }}
                        >
                          {g.items.map((r, i) => {
                            const domain = (r.source || r.domain || 'Web').replace(/^www\./, '');
                            return (
                              <button
                                key={r.url ?? `${g.category}-${i}`}
                                onClick={() => setSelectedResult(r)}
                                className="w-full text-left group rounded-lg border border-white/10 hover:border-white/30 bg-white/5 hover:bg-white/10 transition-colors p-2.5 flex items-start gap-2.5 cursor-pointer"
                              >
                                <div className="w-10 h-10 rounded-md overflow-hidden bg-white/5 border border-white/10 shrink-0 flex items-center justify-center">
                                  {r.thumbnail ? (
                                    <img
                                      src={r.thumbnail}
                                      alt=""
                                      className="w-full h-full object-cover"
                                      onError={e => {
                                        (e.target as HTMLImageElement).style.display = 'none';
                                      }}
                                    />
                                  ) : (
                                    <Globe className="w-4 h-4 text-violet-300/60" />
                                  )}
                                </div>

                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-1.5">
                                    <span className="text-[11px] font-bold text-white truncate">
                                      {domain}
                                    </span>
                                    <span className="shrink-0 text-[8px] font-extrabold uppercase tracking-wider px-1 py-px rounded bg-amber-400/15 text-amber-300 border border-amber-400/30">
                                      Visual
                                    </span>
                                  </div>
                                  {r.title && (
                                    <p className="text-[10px] text-violet-100/60 truncate mt-0.5">
                                      {r.title}
                                    </p>
                                  )}
                                  {r.region && (
                                    <p className="text-[9px] text-violet-100/40 truncate mt-0.5">
                                      {r.region}
                                    </p>
                                  )}
                                </div>

                                <ArrowUpRight className="w-3.5 h-3.5 text-violet-200/30 group-hover:text-violet-200 transition-colors shrink-0 mt-1" />
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Hint */}
          <div className="absolute bottom-3 left-3 text-[10px] text-violet-200/40 font-mono pointer-events-none z-10">
            Click a group header to expand · Click a source to inspect
          </div>

          {/* Inspector popover */}
          {selectedResult && (
            <div className="absolute top-3 right-3 w-80 rounded-xl p-4 border border-violet-400/30 bg-[#1a1530]/95 backdrop-blur-xl shadow-2xl shadow-violet-900/50 space-y-3 z-30">
              <div className="flex items-start justify-between gap-2 border-b border-white/10 pb-2.5">
                <div className="min-w-0">
                  <span className="inline-block px-2 py-0.5 rounded text-[10px] font-bold border mb-1.5 bg-amber-400/15 text-amber-300 border-amber-400/30 uppercase tracking-wider">
                    {matchTypeLabel(selectedResult.match_type) || 'Visual Match'}
                  </span>
                  <h4 className="font-bold text-white text-xs leading-snug line-clamp-2">
                    {selectedResult.title || 'Untitled Source'}
                  </h4>
                </div>
                <button
                  onClick={() => setSelectedResult(null)}
                  className="p-1 text-violet-200/50 hover:text-white hover:bg-white/10 rounded-md cursor-pointer shrink-0"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {selectedResult.thumbnail && (
                <div className="w-full h-28 rounded-lg overflow-hidden border border-white/10 bg-white/5">
                  <img src={selectedResult.thumbnail} alt="" className="w-full h-full object-cover" />
                </div>
              )}

              <div className="space-y-1.5 text-xs">
                <div className="flex items-center gap-2">
                  <Globe className="w-3.5 h-3.5 text-violet-300/60 shrink-0" />
                  <span className="text-violet-200/50">Source:</span>
                  <span className="font-semibold text-white truncate">
                    {selectedResult.source || selectedResult.domain || '—'}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <User className="w-3.5 h-3.5 text-violet-300/60 shrink-0" />
                  <span className="text-violet-200/50">Poster:</span>
                  <span className="font-medium text-violet-100 truncate">
                    {extractPoster(selectedResult)}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <MapPin className="w-3.5 h-3.5 text-violet-300/60 shrink-0" />
                  <span className="text-violet-200/50">Region:</span>
                  <span className="font-medium text-violet-100">
                    {selectedResult.region || 'Unknown'}
                  </span>
                </div>
                {selectedResult.date && (
                  <div className="flex items-center gap-2">
                    <Calendar className="w-3.5 h-3.5 text-violet-300/60 shrink-0" />
                    <span className="text-violet-200/50">Indexed:</span>
                    <span className="font-medium text-violet-100">{selectedResult.date}</span>
                  </div>
                )}
              </div>

              {selectedResult.snippet && (
                <div className="p-2.5 rounded-lg bg-white/5 border border-white/10 text-[11px] text-violet-100/70 leading-relaxed line-clamp-3">
                  {selectedResult.snippet}
                </div>
              )}

              {selectedResult.url && (
                <div className="flex gap-2 pt-1 border-t border-white/10">
                  <a
                    href={selectedResult.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 py-1.5 bg-violet-600 hover:bg-violet-500 text-white rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <ExternalLink className="w-3.5 h-3.5" /> Visit
                  </a>
                  <button
                    onClick={() => copyUrl(selectedResult.url!)}
                    className="px-2.5 py-1.5 bg-white/5 hover:bg-white/10 border border-white/10 text-violet-100 rounded-lg text-xs font-semibold flex items-center gap-1 cursor-pointer"
                  >
                    {copiedUrl ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    {copiedUrl ? 'Copied' : 'Copy'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ─── Timeline sidebar ───────────────────────────────── */}
        <div className="w-56 shrink-0 border-l border-white/10 flex flex-col bg-black/20 backdrop-blur-sm">
          <div className="px-4 pt-4 pb-2 border-b border-white/10">
            <h4 className="text-[11px] font-bold text-violet-200 uppercase tracking-wider flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-violet-300" />
              Timeline
            </h4>
            <p className="text-[9px] text-violet-200/50 mt-0.5">Indexed dates from results</p>
          </div>

          <div className="flex-1 overflow-y-auto propagation-scroll px-4 py-3">
            {timelineItems.length === 0 ? (
              <p className="text-[10px] text-violet-200/40 italic mt-2">
                No date information returned by this search.
              </p>
            ) : (
              <div className="relative ml-2">
                <div className="absolute left-0 top-1 bottom-1 w-px bg-violet-300/20" />
                <div className="space-y-4 pl-5">
                  {timelineItems.map((item, idx) => (
                    <div
                      key={idx}
                      onClick={() => setSelectedResult(item)}
                      className="relative cursor-pointer group"
                    >
                      <div className="absolute -left-5 top-1 w-2.5 h-2.5 rounded-full border-2 border-violet-400 bg-[#1a1530] group-hover:bg-violet-400 transition-colors" />
                      <div className="text-[9px] font-mono text-violet-200/50 leading-tight">
                        {item.date}
                      </div>
                      <div className="text-[10px] font-bold text-white leading-snug truncate mt-0.5">
                        {item.source || item.domain}
                      </div>
                      {item.title && (
                        <div className="text-[9px] text-violet-100/50 truncate leading-tight">
                          {item.title.slice(0, 28)}…
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {results.filter(r => !r.date).length > 0 && (
              <div className="mt-4 pt-3 border-t border-white/10">
                <p className="text-[9px] text-violet-200/40">
                  + {results.filter(r => !r.date).length} source{results.filter(r => !r.date).length !== 1 ? 's' : ''} with no date
                </p>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="px-4 py-3 border-t border-white/10 space-y-1.5">
            <p className="text-[9px] text-violet-200/50 font-bold uppercase tracking-wider">Legend</p>
            <div className="flex items-center gap-2 text-[9px] text-violet-100/60">
              <span className="w-4 h-1.5 bg-violet-400 rounded-sm inline-block" />
              Category ribbon
            </div>
            <div className="flex items-center gap-2 text-[9px] text-violet-100/60">
              <span className="w-3 h-3 rounded-sm inline-block bg-amber-400/20 border border-amber-400/40" />
              Visual match
            </div>
            <div className="flex items-center gap-2 text-[9px] text-violet-100/60">
              <SearchIcon className="w-3 h-3 text-violet-300/60" />
              Click a source to inspect
            </div>
          </div>
        </div>
      </div>

      {/* Scoped scrollbar styling */}
      <style>{`
        .propagation-scroll::-webkit-scrollbar { width: 8px; height: 8px; }
        .propagation-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 4px; }
        .propagation-scroll::-webkit-scrollbar-thumb {
          background: rgba(167,139,250,0.35);
          border-radius: 4px;
        }
        .propagation-scroll::-webkit-scrollbar-thumb:hover {
          background: rgba(167,139,250,0.55);
        }
        .propagation-scroll { scrollbar-width: thin; scrollbar-color: rgba(167,139,250,0.4) transparent; }
      `}</style>
    </div>
  );
};