// ================================================================
// ForensIQ — Propagation Service
//
// Talks to POST /api/propagation/search on the backend.
// The backend handles all SerpApi communication.
// This service NEVER touches the SerpApi key directly.
//
// NOTE: This is completely separate from detectionService.ts.
//       Do NOT import anything from this file in the detection flow.
// ================================================================

// Use relative path — Vite dev proxy forwards /api/* → http://localhost:8000
// In production, the backend serves on the same origin.
const API_BASE = "";

// ----------------------------------------------------------------
// Types
// ----------------------------------------------------------------

export type MatchType = "exact" | "visual" | "possible";

export interface PropagationResult {
  title: string | null;
  url: string | null;
  domain: string | null;
  thumbnail: string | null;
  source: string | null;
  match_type: MatchType;
  snippet: string | null;
  date: string | null;
  region: string | null;
  // Future slots — null until pHash/CLIP is implemented
  phash_verified: null;
  clip_verified: null;
}

export interface PropagationSearchResponse {
  status: "success" | "no_results" | "error";
  total: number;
  exact_matches: PropagationResult[];
  visual_matches: PropagationResult[];
  all_results: PropagationResult[];
  error: string | null;
  message?: string;
}

// ----------------------------------------------------------------
// Loading step labels (shown in the UI during search)
// ----------------------------------------------------------------

export const PROPAGATION_STEPS = [
  "Uploading image to forensic search engine...",
  "Querying Google Lens visual index...",
  "Collecting visual and exact matches...",
  "Processing discovered sources...",
  "Building propagation network...",
] as const;

// ----------------------------------------------------------------
// API call
// ----------------------------------------------------------------

/**
 * Submit an image file for propagation search.
 * Returns structured results from Google Lens via backend.
 *
 * Throws a user-friendly Error on failure.
 */
export async function searchPropagation(
  imageFile: File,
  onStep?: (step: string, index: number) => void
): Promise<PropagationSearchResponse> {
  // Validate on client side before upload
  const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
  if (!allowed.includes(imageFile.type)) {
    throw new Error(
      `Unsupported file type "${imageFile.type}". Please upload a JPG, JPEG, PNG, or WEBP image.`
    );
  }

  const MAX_MB = 20;
  if (imageFile.size > MAX_MB * 1024 * 1024) {
    throw new Error(
      `Image is too large (${(imageFile.size / 1024 / 1024).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`
    );
  }

  // Animate steps (non-blocking, cosmetic)
  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx < PROPAGATION_STEPS.length) {
      onStep?.(PROPAGATION_STEPS[stepIdx], stepIdx);
      stepIdx++;
    }
  }, 900);

  try {
    const formData = new FormData();
    formData.append("file", imageFile, imageFile.name);

    const response = await fetch(`${API_BASE}/api/propagation/search`, {
      method: "POST",
      body: formData,
    });

    const data = (await response.json()) as PropagationSearchResponse;

    if (!response.ok) {
      // Surface the backend's error message when available
      const msg =
        data?.error ||
        `Server returned status ${response.status}.`;
      throw new Error(msg);
    }

    return data;
  } finally {
    clearInterval(stepInterval);
  }
}

// ----------------------------------------------------------------
// Helpers used by UI components
// ----------------------------------------------------------------

/** Returns a human-readable badge label for a match type */
export function matchTypeLabel(type: MatchType | string): string {
  switch (type) {
    case "exact":
      return "EXACT MATCH";
    case "visual":
      return "VISUAL MATCH";
    default:
      return "POSSIBLE MATCH";
  }
}

/** Returns Tailwind colour classes for a match badge */
export function matchTypeBadgeClasses(type: MatchType | string): string {
  switch (type) {
    case "exact":
      return "bg-emerald-50/80 text-emerald-800 border-emerald-200";
    case "visual":
      return "bg-amber-50/80 text-amber-800 border-amber-200";
    default:
      return "bg-stone-100/80 text-stone-700 border-stone-300";
  }
}
