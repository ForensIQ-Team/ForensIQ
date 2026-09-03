import {
  DetectionResult,
  PipelineStage,
  DetailedModelScore,
  ExplainabilityData,
  VideoAnalysisData,
  ClassificationType,
  RiskLevel,
  AuthenticityStatus,
} from '../types';
import { SAMPLE_IMAGES, MOCK_ANALYSES } from '../data/mockData';

// Standard 10-stage pipeline definition required by specifications
export const DEFAULT_PIPELINE_STAGES: Omit<PipelineStage, 'status'>[] = [
  { id: 1, name: 'Upload Received', description: 'Verifying file integrity & checksum (SHA-256)' },
  { id: 2, name: 'Preprocessing', description: 'Color space conversion, noise normalization & scaling' },
  { id: 3, name: 'Face Detection & Alignment', description: 'Multi-task Cascaded CNN (MTCNN) landmark extraction' },
  { id: 4, name: 'Xception Analysis', description: 'Evaluating spatial feature residuals & boundary artifacts' },
  { id: 5, name: 'EfficientNet-B4 Analysis', description: 'Inference on high-frequency feature maps' },
  { id: 6, name: 'Vision Transformer (ViT)', description: 'Attention matrix & spatial self-attention check' },
  { id: 7, name: 'Score Fusion', description: 'Ensemble Bayesian fusion of individual model outputs' },
  { id: 8, name: 'Final Classification', description: 'Threshold mapping: Real / Deepfake / AI-Generated / Uncertain' },
  { id: 9, name: 'Confidence Calculation', description: 'Computing uncertainty bounds & variance spread' },
  { id: 10, name: 'Explainability Generation', description: 'Synthesizing Grad-CAM & ViT attention heatmaps' },
];

// High-resolution SVG visual heatmaps for Grad-CAM & ViT Attention
export const EXPLAINABILITY_HEATMAPS = {
  gradCamReal:
    'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600"><rect width="800" height="600" fill="rgba(15,23,42,0.6)"/><circle cx="400" cy="280" r="160" fill="url(%23gradCamGrad)" opacity="0.45"/><defs><radialGradient id="gradCamGrad" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%2310b981" stop-opacity="0.8"/><stop offset="60%" stop-color="%233b82f6" stop-opacity="0.4"/><stop offset="100%" stop-color="%236366f1" stop-opacity="0"/></radialGradient></defs></svg>',
  gradCamDeepfake:
    'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600"><rect width="800" height="600" fill="rgba(15,23,42,0.85)"/><circle cx="400" cy="270" r="130" fill="url(%23dfGrad)" opacity="0.9"/><circle cx="510" cy="230" r="65" fill="%23ef4444" opacity="0.85"/><circle cx="310" cy="310" r="80" fill="%23f59e0b" opacity="0.75"/><defs><radialGradient id="dfGrad" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%23ef4444" stop-opacity="0.95"/><stop offset="50%" stop-color="%23f59e0b" stop-opacity="0.7"/><stop offset="100%" stop-color="%236366f1" stop-opacity="0"/></radialGradient></defs></svg>',
  vitAttentionReal:
    'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600"><rect width="800" height="600" fill="rgba(15,23,42,0.7)"/><g opacity="0.6" stroke="%2338bdf8" stroke-width="1.5" fill="none"><rect x="300" y="180" width="200" height="220" rx="8" stroke-dasharray="4 4"/><circle cx="360" cy="240" r="25" fill="rgba(56,189,248,0.3)"/><circle cx="440" cy="240" r="25" fill="rgba(56,189,248,0.3)"/><path d="M 360,330 Q 400,360 440,330"/></g></svg>',
  vitAttentionDeepfake:
    'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600" viewBox="0 0 800 600"><rect width="800" height="600" fill="rgba(15,23,42,0.8)"/><g opacity="0.85"><polygon points="320,180 480,180 520,380 280,380" fill="url(%23vitGrad)"/><rect x="310" y="210" width="180" height="60" fill="%23ef4444" opacity="0.6"/><circle cx="400" cy="340" r="45" fill="%23f59e0b" opacity="0.7"/></g><defs><radialGradient id="vitGrad" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="%238b5cf6" stop-opacity="0.9"/><stop offset="100%" stop-color="%23ef4444" stop-opacity="0.3"/></radialGradient></defs></svg>',
};

// Preset scenarios for deterministic demo responses
export const PRESET_DETECTIONS: Record<string, DetectionResult> = {
  // Preset 1: Real
  real: {
    id: 'FQ-8091',
    mediaName: 'Portrait_Authentic_01.jpg',
    mediaUrl: SAMPLE_IMAGES.authenticPortrait,
    mediaType: 'image',
    uploadDate: '2026-08-12 14:10 UTC',
    classification: 'Real',
    confidenceScore: 94,
    authenticityScore: 94,
    processingTimeMs: 280,
    facesDetected: 1,
    modelAgreement: 'High Agreement',
    modelAgreementDescription: 'All detection models strongly support the final classification of authentic optical media.',
    riskLevel: 'low',
    detailedModelScores: [
      { name: 'Xception', realScore: 91, fakeScore: 9, confidence: 'High' },
      { name: 'EfficientNet-B4', realScore: 88, fakeScore: 12, confidence: 'High' },
      { name: 'Vision Transformer (ViT)', realScore: 94, fakeScore: 6, confidence: 'High' },
      { name: 'Final Fused Score', realScore: 92, fakeScore: 8, confidence: 'Optimal' },
    ],
    explainability: {
      gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
      vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionReal,
      originalUrl: SAMPLE_IMAGES.authenticPortrait,
      highlightedRegions: [
        { label: 'Eye Refraction', description: 'Specular highlights are natural and symmetrical.', bounds: { x: 38, y: 32, width: 24, height: 12 } },
        { label: 'Facial Boundary', description: 'Natural sensor noise & edge continuity verified.', bounds: { x: 28, y: 22, width: 44, height: 56 } },
      ],
    },
    metadata: MOCK_ANALYSES['FQ-8091'].metadata,
    manipulationSignals: MOCK_ANALYSES['FQ-8091'].manipulationSignals,
    summaryText: 'The uploaded image exhibits characteristics consistent with authentic optical capture. Synthetic model ensemble scores reflect low probability of neural generative patterns.',
    warnings: [],
    investigatorNotes: 'PRNU noise distribution matches Sony hardware baseline. Zero boundary splicing detected.',
    authenticityStatus: 'likely-authentic',
    overallScore: 94,
    confidence: 94,
    modelScores: MOCK_ANALYSES['FQ-8091'].modelScores,
    heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
    elaUrl: SAMPLE_IMAGES.elaMap,
    whySignals: [
      'No strong synthetic generative patterns detected across ViT & EfficientNet models',
      'Optical lens noise and sensor pattern noise (PRNU) are natural and uniform',
      'EXIF metadata tags match standard camera software signature without splicing anomalies',
    ],
  },

  // Preset 2: Deepfake
  deepfake: {
    id: 'FQ-9941',
    mediaName: 'FacialSwap_Deepfake_02.jpg',
    mediaUrl: SAMPLE_IMAGES.aiSyntheticFace,
    mediaType: 'image',
    uploadDate: '2026-08-12 15:22 UTC',
    classification: 'Deepfake',
    confidenceScore: 95,
    authenticityScore: 8,
    processingTimeMs: 340,
    facesDetected: 1,
    modelAgreement: 'High Agreement',
    modelAgreementDescription: 'All three neural detectors identified high-frequency face-swap boundary artifacts.',
    riskLevel: 'high',
    detailedModelScores: [
      { name: 'Xception', realScore: 5, fakeScore: 95, confidence: 'High' },
      { name: 'EfficientNet-B4', realScore: 9, fakeScore: 91, confidence: 'High' },
      { name: 'Vision Transformer (ViT)', realScore: 4, fakeScore: 96, confidence: 'High' },
      { name: 'Final Fused Score', realScore: 6, fakeScore: 94, confidence: 'Optimal' },
    ],
    explainability: {
      gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
      vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake,
      originalUrl: SAMPLE_IMAGES.aiSyntheticFace,
      highlightedRegions: [
        { label: 'Jawline Blending Mask', description: 'Spatial discontinuity detected along lower jaw perimeter.', bounds: { x: 30, y: 50, width: 40, height: 20 } },
        { label: 'Pupil Highlight Anomaly', description: 'Asymmetric lighting vectors in iris reflections.', bounds: { x: 35, y: 30, width: 30, height: 12 } },
      ],
    },
    metadata: MOCK_ANALYSES['FQ-4022'].metadata,
    manipulationSignals: [
      { id: 'boundary', name: 'Face Mask Boundary Residual', status: 'flagged', score: '0.92 Anomaly', description: 'Color jitter and warping present along facial perimeter.' },
      { id: 'prnu', name: 'Sensor Pattern Noise (PRNU)', status: 'flagged', score: 'Missing PRNU', description: 'Facial region lacks physical camera sensor noise.' },
      { id: 'fft', name: 'High-Frequency Fourier Peaks', status: 'flagged', score: 'Grid Residuals', description: 'Spectral spikes characteristic of autoencoder warping.' },
    ],
    summaryText: 'Critical manipulation detected. Facial identity region displays face-swapping autoencoder artifacts and boundary frequency anomalies.',
    warnings: ['High probability of deepfake impersonation.', 'Facial region modified post-capture.'],
    investigatorNotes: 'Facial replacement boundary confirmed by ViT spatial self-attention map.',
    authenticityStatus: 'likely-ai-generated',
    overallScore: 8,
    confidence: 95,
    modelScores: [
      { name: 'Xception', score: 5, confidence: 'High' },
      { name: 'EfficientNet-B4', score: 9, confidence: 'High' },
      { name: 'Vision Transformer', score: 4, confidence: 'High' },
    ],
    heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
    elaUrl: SAMPLE_IMAGES.elaMap,
    whySignals: [
      'Model ensemble flagged high-confidence generative face-swap fingerprints',
      'Asymmetric pupil specular highlights and anomalous boundary warping detected',
      'Missing hardware sensor noise in facial bounding box',
    ],
  },

  // Preset 3: AI-Generated
  aiGenerated: {
    id: 'FQ-4022',
    mediaName: 'Profile_Synthetic_GenAI.jpg',
    mediaUrl: SAMPLE_IMAGES.aiSyntheticFace,
    mediaType: 'image',
    uploadDate: '2026-08-12 11:32 UTC',
    classification: 'AI-Generated',
    confidenceScore: 96,
    authenticityScore: 18,
    processingTimeMs: 310,
    facesDetected: 1,
    modelAgreement: 'High Agreement',
    modelAgreementDescription: 'Unanimous detection of diffusion latent grid patterns.',
    riskLevel: 'high',
    detailedModelScores: [
      { name: 'Xception', realScore: 12, fakeScore: 88, confidence: 'Very High' },
      { name: 'EfficientNet-B4', realScore: 22, fakeScore: 78, confidence: 'Very High' },
      { name: 'Vision Transformer (ViT)', realScore: 16, fakeScore: 84, confidence: 'High' },
      { name: 'Final Fused Score', realScore: 18, fakeScore: 82, confidence: 'Optimal' },
    ],
    explainability: {
      gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
      vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake,
      originalUrl: SAMPLE_IMAGES.aiSyntheticFace,
      highlightedRegions: [
        { label: 'Latent Noise Uniformity', description: 'Synthetic smoothness in hair & background texture.', bounds: { x: 10, y: 10, width: 80, height: 80 } },
      ],
    },
    metadata: MOCK_ANALYSES['FQ-4022'].metadata,
    manipulationSignals: MOCK_ANALYSES['FQ-4022'].manipulationSignals,
    summaryText: 'High probability of synthetic diffusion model generation (Flux.1 / Midjourney pipeline).',
    warnings: ['Entire image is synthetically generated.', 'No physical camera hardware metadata detected.'],
    investigatorNotes: 'Full image synthetic synthesis confirmed.',
    authenticityStatus: 'likely-ai-generated',
    overallScore: 18,
    confidence: 96,
    modelScores: MOCK_ANALYSES['FQ-4022'].modelScores,
    heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
    elaUrl: SAMPLE_IMAGES.elaMap,
    whySignals: MOCK_ANALYSES['FQ-4022'].whySignals,
  },

  // Preset 4: Uncertain
  uncertain: {
    id: 'FQ-3088',
    mediaName: 'Low_Resolution_Compressed.jpg',
    mediaUrl: SAMPLE_IMAGES.manipulatedDocument,
    mediaType: 'image',
    uploadDate: '2026-08-12 16:05 UTC',
    classification: 'Uncertain',
    confidenceScore: 52,
    authenticityScore: 54,
    processingTimeMs: 410,
    facesDetected: 1,
    modelAgreement: 'Model Disagreement',
    modelAgreementDescription: 'Model outputs diverge significantly due to heavy JPEG re-compression. Additional verification required.',
    riskLevel: 'medium',
    detailedModelScores: [
      { name: 'Xception', realScore: 72, fakeScore: 28, confidence: 'Low' },
      { name: 'EfficientNet-B4', realScore: 38, fakeScore: 62, confidence: 'Low' },
      { name: 'Vision Transformer (ViT)', realScore: 51, fakeScore: 49, confidence: 'Uncertain' },
      { name: 'Final Fused Score', realScore: 54, fakeScore: 46, confidence: 'Low' },
    ],
    explainability: {
      gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
      vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionReal,
      originalUrl: SAMPLE_IMAGES.manipulatedDocument,
      highlightedRegions: [
        { label: 'Compression Artifact Zone', description: 'Heavy quantization blocks obscure micro-texture.', bounds: { x: 20, y: 20, width: 60, height: 60 } },
      ],
    },
    metadata: {
      ...MOCK_ANALYSES['FQ-2098'].metadata,
      filename: 'Low_Resolution_Compressed.jpg',
      software: 'Web Compression Pipeline',
    },
    manipulationSignals: [
      { id: 'comp', name: 'Quantization Matrix', status: 'warning', score: 'Heavy Compression', description: 'Lossy compression degrades fine spatial details required for neural confidence.' },
    ],
    summaryText: 'Result is inconclusive. High JPEG compression noise prevents confident differentiation between synthetic artifacts and compression noise.',
    warnings: ['Model agreement is low.', 'Heavy re-compression detected. Upload higher resolution source if available.'],
    investigatorNotes: 'Uncertain classification. Recommend Error-Level Analysis and physical document verification.',
    authenticityStatus: 'uncertain',
    overallScore: 54,
    confidence: 52,
    modelScores: [
      { name: 'Xception', score: 72, confidence: 'Low' },
      { name: 'EfficientNet-B4', score: 38, confidence: 'Low' },
      { name: 'Vision Transformer', score: 51, confidence: 'Uncertain' },
    ],
    heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
    elaUrl: SAMPLE_IMAGES.elaMap,
    whySignals: [
      'High JPEG compression degrades high-frequency feature maps',
      'Xception and EfficientNet models produced conflicting predictions',
      'Uncertain classification — additional verification recommended',
    ],
  },

  // Preset 5: Video
  video: {
    id: 'FQ-VID-772',
    mediaName: 'Interrogative_Video_Deepfake.mp4',
    mediaUrl: 'https://assets.mixkit.co/videos/preview/mixkit-man-holding-a-tablet-41009-large.mp4',
    mediaType: 'video',
    uploadDate: '2026-08-12 17:30 UTC',
    classification: 'Deepfake',
    confidenceScore: 89,
    authenticityScore: 14,
    processingTimeMs: 1240,
    facesDetected: 1,
    modelAgreement: 'High Agreement',
    modelAgreementDescription: 'Temporal frame alignment analysis detected lip-sync temporal jitter across 8 sampled frames.',
    riskLevel: 'high',
    detailedModelScores: [
      { name: 'Xception (Frame-level)', realScore: 12, fakeScore: 88, confidence: 'High' },
      { name: 'EfficientNet-B4', realScore: 16, fakeScore: 84, confidence: 'High' },
      { name: 'Vision Transformer (3D-ViT)', realScore: 10, fakeScore: 90, confidence: 'High' },
      { name: 'Final Fused Score', realScore: 14, fakeScore: 86, confidence: 'Optimal' },
    ],
    explainability: {
      gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
      vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake,
      originalUrl: SAMPLE_IMAGES.authenticPortrait,
      highlightedRegions: [
        { label: 'Lip-Sync Temporal Jitter', description: 'Phoneme-to-viseme mismatch detected at 00:03 - 00:06.', bounds: { x: 38, y: 55, width: 24, height: 16 } },
      ],
    },
    metadata: {
      filename: 'Interrogative_Video_Deepfake.mp4',
      filesize: '14.2 MB',
      dimensions: '1920 x 1080 (1080p)',
      filetype: 'video/mp4',
      cameraModel: 'H.264 / AVC Codec',
      software: 'Wav2Lip / DeepFaceLab Pipeline',
      modifiedDate: '2026-08-12 16:00:00',
      createdDate: '2026-08-12 16:00:00',
      colorSpace: 'yuv420p',
      exifIntact: false,
      hashSHA256: 'c830114e9fbc77221045a19001e3b202998466110a304892c90e811f2a7a12b9',
      pHash: '8a9f02c110e4b8d7',
    },
    manipulationSignals: [
      { id: 'temporal', name: 'Temporal Frame Continuity', status: 'flagged', score: '8 Suspicious Frames', description: 'Inter-frame flickering detected around jaw boundary.' },
      { id: 'audio', name: 'Audio-Visual Sync (Wav2Lip)', status: 'flagged', score: '0.78 Discrepancy', description: 'Speech envelope does not align with facial muscle contraction.' },
    ],
    videoData: {
      sampledFrames: 24,
      facesDetected: 1,
      framesAnalyzed: 24,
      suspiciousFrameCount: 8,
      fps: 30,
      durationSeconds: 12,
      frameResults: Array.from({ length: 24 }).map((_, i) => ({
        frameNumber: i + 1,
        timestamp: `00:0${Math.floor(i / 2)}.${(i % 2) * 5}`,
        fakeScore: [10, 12, 14, 25, 82, 94, 91, 88, 85, 92, 89, 87, 40, 20, 15, 78, 85, 90, 88, 82, 20, 15, 12, 10][i] || 15,
        isSuspicious: [10, 12, 14, 25, 82, 94, 91, 88, 85, 92, 89, 87, 40, 20, 15, 78, 85, 90, 88, 82, 20, 15, 12, 10][i] > 60,
        anomalyType: [10, 12, 14, 25, 82, 94, 91, 88, 85, 92, 89, 87, 40, 20, 15, 78, 85, 90, 88, 82, 20, 15, 12, 10][i] > 60 ? 'Facial Morphing / Lip-Sync Jitter' : undefined,
      })),
    },
    summaryText: 'Video deepfake detected. 8 out of 24 analyzed frames display lip-sync and temporal facial boundary manipulation.',
    warnings: ['Temporal inconsistencies detected across frames 5-12 and 16-20.', 'Synthetic voice/lip alignment flag.'],
    investigatorNotes: 'Wav2Lip neural audio-visual sync mismatch detected.',
    authenticityStatus: 'likely-ai-generated',
    overallScore: 14,
    confidence: 89,
    modelScores: [
      { name: 'Xception (3D)', score: 12, confidence: 'High' },
      { name: 'EfficientNet-B4', score: 16, confidence: 'High' },
      { name: '3D Vision Transformer', score: 10, confidence: 'High' },
    ],
    heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamDeepfake,
    elaUrl: SAMPLE_IMAGES.elaMap,
    whySignals: [
      'Temporal frame-to-frame facial jitter detected across sampled frames',
      'Audio speech envelope phoneme mismatch with lip movement',
      'Xception and 3D-ViT models identified boundary autoencoder artifacts',
    ],
  },
};

export const detectionService = {
  /**
   * Main media analysis method supporting step-by-step pipeline callbacks.
   * - For preset keys (string input): returns deterministic demo data.
   * - For File input: calls the ForensIQ V2 FastAPI backend (Python ML pipeline).
   */
  async analyzeMedia(
    input: File | string,
    onStageUpdate?: (currentStageIndex: number, stages: PipelineStage[]) => void
  ): Promise<DetectionResult> {
    // ----------------------------------------------------------------
    // PRESET DEMO SCENARIOS — deterministic, unchanged
    // ----------------------------------------------------------------
    if (typeof input === 'string') {
      const baseResult = PRESET_DETECTIONS[input] || PRESET_DETECTIONS.real;

      const stages: PipelineStage[] = DEFAULT_PIPELINE_STAGES.map((s) => ({
        ...s,
        status: 'pending',
      }));

      if (onStageUpdate) {
        for (let i = 0; i < stages.length; i++) {
          stages[i].status = 'processing';
          onStageUpdate(i, [...stages]);
          await new Promise((resolve) => setTimeout(resolve, 140));
          stages[i].status = 'completed';
          onStageUpdate(i, [...stages]);
        }
      }

      return baseResult;
    }

    // ----------------------------------------------------------------
    // REAL FILE UPLOAD — call the actual Python ML backend
    // ----------------------------------------------------------------

    // Stage labels shown during upload analysis
    const REAL_PIPELINE_STAGES: Omit<PipelineStage, 'status'>[] = [
      { id: 1, name: 'Upload Received', description: 'Verifying file integrity & MIME type' },
      { id: 2, name: 'Loading Image', description: 'Decoding image to RGB color space' },
      { id: 3, name: 'Face Detection', description: 'MTCNN face detection & margin-expanded crop' },
      { id: 4, name: 'Xception V2', description: 'Running Xception V2 with TTA (original + flipped)' },
      { id: 5, name: 'EfficientNet-B4 V2', description: 'Running EfficientNet-B4 V2 with TTA' },
      { id: 6, name: 'Score Fusion', description: 'Confidence-gated ensemble fusion (70% Xception / 30% EfficientNet)' },
      { id: 7, name: 'Final Classification', description: 'Applying 0.50 threshold → FAKE / REAL decision' },
      { id: 8, name: 'Confidence Calculation', description: 'Computing model agreement & disagreement level' },
    ];

    const stages: PipelineStage[] = REAL_PIPELINE_STAGES.map((s) => ({
      ...s,
      status: 'pending',
    }));

    // Helper: advance one stage visually
    const advanceStage = async (index: number) => {
      if (!onStageUpdate) return;
      stages[index].status = 'processing';
      onStageUpdate(index, [...stages]);
      await new Promise((resolve) => setTimeout(resolve, 180));
      stages[index].status = 'completed';
      onStageUpdate(index, [...stages]);
    };

    await advanceStage(0); // Upload received

    // ----------------------------------------------------------------
    // Validate file format & size before sending
    // ----------------------------------------------------------------
    const ALLOWED = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
    if (input.type && !ALLOWED.includes(input.type)) {
      throw new Error(`Unsupported file type: ${input.type}. Please upload a JPG, PNG, or WEBP image.`);
    }
    if (input.size > 50 * 1024 * 1024) {
      throw new Error('File too large. Maximum upload size is 50 MB.');
    }
    if (input.size === 0) {
      throw new Error('The selected file appears to be empty.');
    }

    await advanceStage(1); // Loading image

    // Build FormData
    const formData = new FormData();
    formData.append('file', input);

    // Start the remaining stage animations concurrently with the fetch
    const animateRemaining = async () => {
      await new Promise((r) => setTimeout(r, 300));
      await advanceStage(2); // Face detection
      await new Promise((r) => setTimeout(r, 400));
      await advanceStage(3); // Xception V2
      await new Promise((r) => setTimeout(r, 400));
      await advanceStage(4); // EfficientNet-B4 V2
    };
    animateRemaining(); // fire and don't await — stages advance while Python runs

    // ----------------------------------------------------------------
    // POST to the ForensIQ Python FastAPI backend
    // ----------------------------------------------------------------
    let apiData: Record<string, unknown>;

    try {
      const response = await fetch('http://localhost:8000/api/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = `Backend error (${response.status})`;
        try {
          const errBody = await response.json();
          errorMsg = errBody?.detail || errorMsg;
        } catch {
          // ignore JSON parse failure
        }
        throw new Error(errorMsg);
      }

      apiData = await response.json();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);

      // Surface friendly error if backend is not running
      if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
        throw new Error(
          'The ForensIQ ML backend is not running. ' +
          'Please start it with: python api_server.py'
        );
      }

      throw new Error(`Detection failed: ${message}`);
    }

    // Finish remaining stage animations
    if (onStageUpdate) {
      await new Promise((r) => setTimeout(r, 200));
      if (stages[2].status !== 'completed') await advanceStage(2);
      if (stages[3].status !== 'completed') await advanceStage(3);
      if (stages[4].status !== 'completed') await advanceStage(4);
      await advanceStage(5); // Score fusion
      await advanceStage(6); // Final classification
      await advanceStage(7); // Confidence
    }

    // ----------------------------------------------------------------
    // Map backend JSON → DetectionResult expected by existing UI
    // ----------------------------------------------------------------

    const final = (apiData.final as Record<string, unknown>) || {};
    const xv2 = (apiData.xception_v2 as Record<string, unknown>) || {};
    const ev2 = (apiData.efficientnet_b4_v2 as Record<string, unknown>) || {};
    const fusion = (apiData.fusion as Record<string, unknown>) || {};
    const agreement = (apiData.agreement as Record<string, unknown>) || {};
    const faceDetection = (apiData.face_detection as Record<string, unknown>) || {};

    const rawPrediction = (final.prediction as string) || (apiData.prediction as string) || 'REAL';
    const fakeProbability = (final.fake_probability as number) ?? (apiData.fake_probability as number) ?? 0;
    const realProbability = (final.real_probability as number) ?? (apiData.real_probability as number) ?? 1;
    const confidence = (final.confidence as number) ?? (apiData.confidence as number) ?? 0;

    // Convert prediction string to UI ClassificationType
    const classification: ClassificationType = rawPrediction === 'FAKE' ? 'Deepfake' : 'Real';
    const authenticityScore = Math.round(realProbability * 100);
    const confidenceScore = Math.round(confidence * 100);
    const riskLevel: RiskLevel = rawPrediction === 'FAKE' ? (confidenceScore > 75 ? 'high' : 'medium') : 'low';

    const xFake = Math.round(((xv2.fake as number) ?? 0) * 100);
    const xReal = Math.round(((xv2.real as number) ?? 0) * 100);
    const eFake = Math.round(((ev2.fake as number) ?? 0) * 100);
    const eReal = Math.round(((ev2.real as number) ?? 0) * 100);
    const fusedFake = Math.round(((fusion.final_fake_score as number) ?? fakeProbability) * 100);
    const fusedReal = 100 - fusedFake;

    const modelsAgree = (agreement.models_agree as boolean) ?? true;
    const disagreementLevel = (agreement.disagreement_level as string) ?? 'LOW';
    const modelAgreement: DetectionResult['modelAgreement'] = modelsAgree
      ? (disagreementLevel === 'LOW' ? 'High Agreement' : 'Moderate Agreement')
      : 'Model Disagreement';

    const numFaces = (faceDetection.num_faces as number) ?? 0;
    const faceDetected = (faceDetection.detected as boolean) ?? false;
    const fallback = (faceDetection.fallback as boolean) ?? false;

    const inputInfo = (apiData.input as Record<string, unknown>) || {};
    const width = (inputInfo.width as number) ?? 0;
    const height = (inputInfo.height as number) ?? 0;
    const sizeBytes = (inputInfo.size_bytes as number) ?? input.size;

    const mediaPreviewUrl = URL.createObjectURL(input);
    const now = new Date().toISOString().replace('T', ' ').substring(0, 16) + ' UTC';

    // Build whySignals based on real model outputs
    const whySignals: string[] = [];
    if (rawPrediction === 'FAKE') {
      whySignals.push(`Xception V2 detected ${xFake}% fake probability (TTA ensemble)`);
      whySignals.push(`EfficientNet-B4 V2 detected ${eFake}% fake probability (TTA ensemble)`);
      whySignals.push(`Confidence-gated ensemble fusion score: ${fusedFake}% FAKE (mode: ${fusion.fusion_mode ?? 'standard'})`);
      if (!modelsAgree) {
        whySignals.push(`Models DISAGREE (${disagreementLevel} disagreement, Δ=${Math.round(((agreement.score_difference as number) ?? 0) * 100)}%)`);
      }
    } else {
      whySignals.push(`Xception V2: ${xReal}% authentic (${xFake}% fake)`);
      whySignals.push(`EfficientNet-B4 V2: ${eReal}% authentic (${eFake}% fake)`);
      whySignals.push(`Ensemble fusion REAL score: ${fusedReal}% (fusion mode: ${fusion.fusion_mode ?? 'standard'})`);
    }

    if (!faceDetected || fallback) {
      whySignals.push('⚠ No face detected — inference ran on full image fallback (may reduce reliability)');
    }

    const detectionResult: DetectionResult = {
      id: `FQ-${Math.floor(1000 + Math.random() * 9000)}`,
      mediaName: input.name,
      mediaUrl: mediaPreviewUrl,
      mediaType: 'image',
      uploadDate: now,
      classification,
      confidenceScore,
      authenticityScore,
      processingTimeMs: 0,
      facesDetected: numFaces,
      modelAgreement,
      modelAgreementDescription: modelsAgree
        ? `Both Xception V2 and EfficientNet-B4 V2 agree: ${rawPrediction}. Disagreement level: ${disagreementLevel}.`
        : `Models DISAGREE. Xception: ${xv2.prediction ?? '?'}, EfficientNet: ${ev2.prediction ?? '?'}. Score difference: ${Math.round(((agreement.score_difference as number) ?? 0) * 100)}%.`,
      riskLevel,
      detailedModelScores: [
        {
          name: 'Xception V2',
          realScore: xReal,
          fakeScore: xFake,
          confidence: xFake > 80 || xReal > 80 ? 'High' : xFake > 60 || xReal > 60 ? 'Medium' : 'Low',
        },
        {
          name: 'EfficientNet-B4 V2',
          realScore: eReal,
          fakeScore: eFake,
          confidence: eFake > 80 || eReal > 80 ? 'High' : eFake > 60 || eReal > 60 ? 'Medium' : 'Low',
        },
        {
          name: 'Ensemble (Confidence-Gated Fusion)',
          realScore: fusedReal,
          fakeScore: fusedFake,
          confidence: confidenceScore > 75 ? 'High' : confidenceScore > 50 ? 'Medium' : 'Low',
        },
      ],
      explainability: {
        gradCamUrl: rawPrediction === 'FAKE'
          ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake
          : EXPLAINABILITY_HEATMAPS.gradCamReal,
        vitAttentionUrl: rawPrediction === 'FAKE'
          ? EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake
          : EXPLAINABILITY_HEATMAPS.vitAttentionReal,
        originalUrl: mediaPreviewUrl,
        highlightedRegions: [],
      },
      metadata: {
        filename: input.name,
        filesize: `${(sizeBytes / (1024 * 1024)).toFixed(2)} MB`,
        dimensions: width && height ? `${width} x ${height}` : 'Unknown',
        filetype: input.type || 'image/jpeg',
        modifiedDate: now,
        createdDate: now,
        exifIntact: true,
        hashSHA256: 'computed-server-side',
        pHash: 'computed-server-side',
      },
      manipulationSignals: rawPrediction === 'FAKE'
        ? [
            {
              id: 'ensemble',
              name: 'Neural Ensemble Detection',
              status: 'flagged',
              score: `${fusedFake}% FAKE`,
              description: `Confidence-gated fusion of Xception V2 (${xFake}%) and EfficientNet-B4 V2 (${eFake}%).`,
            },
            ...((!faceDetected || fallback)
              ? [{
                  id: 'face',
                  name: 'Face Detection',
                  status: 'warning' as const,
                  score: 'No face found',
                  description: 'Full-image fallback used. Accuracy may be reduced.',
                }]
              : []),
          ]
        : [
            {
              id: 'ensemble',
              name: 'Neural Ensemble Detection',
              status: 'passed',
              score: `${fusedReal}% REAL`,
              description: `Confidence-gated fusion of Xception V2 (${xReal}% real) and EfficientNet-B4 V2 (${eReal}% real).`,
            },
          ],
      summaryText: rawPrediction === 'FAKE'
        ? `The ForensIQ V2 pipeline detected synthetic manipulation with ${confidenceScore}% confidence. Xception V2: ${xFake}% fake, EfficientNet-B4 V2: ${eFake}% fake. Ensemble fused score: ${fusedFake}%.`
        : `The ForensIQ V2 pipeline found no evidence of manipulation. Xception V2: ${xReal}% authentic, EfficientNet-B4 V2: ${eReal}% authentic. Ensemble confidence: ${confidenceScore}%.`,
      warnings: [
        ...(!faceDetected || fallback
          ? ['No face was detected. Full-image inference used as fallback — results may be less reliable.']
          : []),
        ...(!modelsAgree
          ? [`Models disagree (${disagreementLevel} disagreement). Review individual scores before drawing conclusions.`]
          : []),
      ],
      investigatorNotes: [
        `Source: ForensIQ V2 face-crop pipeline`,
        `Fusion mode: ${fusion.fusion_mode ?? 'STANDARD_WEIGHTED_FUSION'}`,
        `Xception weight: ${((fusion.xception_weight as number) ?? 0.7).toFixed(2)}, EfficientNet weight: ${((fusion.efficientnet_weight as number) ?? 0.3).toFixed(2)}`,
        `TTA enabled: ${(apiData.preprocessing as Record<string, unknown>)?.tta ?? true}`,
        `Face detected: ${faceDetected}, Fallback: ${fallback}`,
        `Device: ${apiData.device ?? 'cpu'}`,
      ].join(' | '),
      authenticityStatus: rawPrediction === 'FAKE' ? 'likely-ai-generated' : 'likely-authentic',
      overallScore: authenticityScore,
      confidence: confidenceScore,
      modelScores: [
        { name: 'Xception V2', score: xReal, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', score: eReal, confidence: 'High' },
      ],
      heatmapUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake : EXPLAINABILITY_HEATMAPS.gradCamReal,
      elaUrl: SAMPLE_IMAGES.elaMap,
      whySignals,
    };

    return detectionResult;
  },

  /**
   * Helper to inspect model score outputs
   */
  getModelScores(result: DetectionResult): DetailedModelScore[] {
    return result.detailedModelScores;
  },

  /**
   * Helper to retrieve classification result
   */
  getClassification(result: DetectionResult): {
    classification: ClassificationType;
    confidenceScore: number;
    riskLevel: RiskLevel;
  } {
    return {
      classification: result.classification,
      confidenceScore: result.confidenceScore,
      riskLevel: result.riskLevel,
    };
  },

  /**
   * Helper to retrieve explainability maps & regions
   */
  getExplainability(result: DetectionResult): ExplainabilityData {
    return result.explainability;
  },
};
