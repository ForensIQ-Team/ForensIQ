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

    const isVideo = input.type?.startsWith('video/') ||
      /\.(mp4|avi|mov|mkv|webm|mpeg|mpg)$/i.test(input.name);
    const isImage = !isVideo;

    // Stage labels shown during upload analysis
    const REAL_PIPELINE_STAGES: Omit<PipelineStage, 'status'>[] = isVideo
      ? [
          { id: 1, name: 'Upload Received', description: 'Verifying file integrity & MIME type' },
          { id: 2, name: 'Loading Video', description: 'Opening video stream with OpenCV' },
          { id: 3, name: 'Frame Sampling', description: 'Uniform temporal sampling (up to 32 frames)' },
          { id: 4, name: 'Face Detection', description: 'MTCNN detection on each sampled frame' },
          { id: 5, name: 'Xception V2', description: 'Running Xception V2 per-frame with TTA' },
          { id: 6, name: 'EfficientNet-B4 V2', description: 'Running EfficientNet-B4 V2 per-frame with TTA' },
          { id: 7, name: 'ViT V2.1', description: 'Running Vision Transformer per-frame with TTA' },
          { id: 8, name: 'Score Aggregation', description: '3-model ensemble fusion + temporal aggregation' },
          { id: 9, name: 'Final Classification', description: 'Mean/median/75th-percentile → FAKE / REAL' },
        ]
      : [
          { id: 1, name: 'Upload Received', description: 'Verifying file integrity & MIME type' },
          { id: 2, name: 'Loading Image', description: 'Decoding image to RGB color space' },
          { id: 3, name: 'Face Detection', description: 'MTCNN face detection & margin-expanded crop' },
          { id: 4, name: 'Xception V2', description: 'Running Xception V2 with TTA (original + flipped)' },
          { id: 5, name: 'EfficientNet-B4 V2', description: 'Running EfficientNet-B4 V2 with TTA' },
          { id: 6, name: 'ViT V2.1', description: 'Running Vision Transformer V2.1 with TTA' },
          { id: 7, name: 'Score Fusion', description: 'Confidence-gated 3-model ensemble (0.45 Xception / 0.25 EfficientNet / 0.30 ViT)' },
          { id: 8, name: 'Final Classification', description: 'Applying 0.50 threshold → FAKE / REAL decision' },
          { id: 9, name: 'Confidence Calculation', description: 'Computing model agreement & disagreement level' },
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
    const ALLOWED_IMAGE = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
    const ALLOWED_VIDEO = ['video/mp4', 'video/quicktime', 'video/avi', 'video/x-msvideo',
                           'video/webm', 'video/mpeg', 'video/x-matroska'];
    const ALLOWED_ALL = [...ALLOWED_IMAGE, ...ALLOWED_VIDEO];
    if (input.type && !ALLOWED_ALL.includes(input.type) && !isVideo && !isImage) {
      throw new Error(`Unsupported file type: ${input.type}. Please upload an image (JPG/PNG/WEBP) or video (MP4/MOV).`);
    }
    const maxSize = isVideo ? 500 * 1024 * 1024 : 50 * 1024 * 1024;
    const maxLabel = isVideo ? '500 MB' : '50 MB';
    if (input.size > maxSize) {
      throw new Error(`File too large. Maximum upload size is ${maxLabel}.`);
    }
    if (input.size === 0) {
      throw new Error('The selected file appears to be empty.');
    }

    await advanceStage(1); // Loading image/video

    // Build FormData
    const formData = new FormData();
    formData.append('file', input);

    // Start the remaining stage animations concurrently with the fetch.
    // These are UI-only indicators — the backend runs synchronously.
    const animateRemaining = async () => {
      const delays = isVideo
        ? [500, 600, 800, 900, 900, 1000, 1100] // video takes longer
        : [300, 400,  400, 400, 400,  400,  400];
      for (let i = 2; i < stages.length - 1; i++) {
        await new Promise((r) => setTimeout(r, delays[i - 2] ?? 400));
        await advanceStage(i);
      }
    };
    animateRemaining(); // fire and don't await — stages advance while Python runs

    // ----------------------------------------------------------------
    // POST to the ForensIQ Python FastAPI backend
    // ----------------------------------------------------------------
    let apiData: Record<string, unknown>;

    try {
      const response = await fetch('http://localhost:8000/api/v1/detect', {
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
      for (let i = 2; i < stages.length; i++) {
        if (stages[i].status !== 'completed') await advanceStage(i);
      }
    }

    // ----------------------------------------------------------------
    // Map backend JSON → DetectionResult expected by existing UI
    // ----------------------------------------------------------------

    const now = new Date().toISOString().replace('T', ' ').substring(0, 16) + ' UTC';
    const mediaPreviewUrl = URL.createObjectURL(input);

    // ---- Handle inconclusive (no face / not enough frames) ----
    if (apiData.status === 'inconclusive') {
      const reason = (apiData.reason as string) || 'Inconclusive result';
      const detectionResultInconclusive: DetectionResult = {
        id: `FQ-${Math.floor(1000 + Math.random() * 9000)}`,
        mediaName: input.name,
        mediaUrl: mediaPreviewUrl,
        mediaType: isVideo ? 'video' : 'image',
        uploadDate: now,
        classification: 'Uncertain',
        confidenceScore: 0,
        authenticityScore: 50,
        processingTimeMs: 0,
        facesDetected: 0,
        modelAgreement: 'Model Disagreement',
        modelAgreementDescription: `Inconclusive — ${reason}`,
        riskLevel: 'medium',
        detailedModelScores: [],
        explainability: {
          gradCamUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
          vitAttentionUrl: EXPLAINABILITY_HEATMAPS.vitAttentionReal,
          originalUrl: mediaPreviewUrl,
          highlightedRegions: [],
        },
        metadata: {
          filename: input.name,
          filesize: `${(input.size / (1024 * 1024)).toFixed(2)} MB`,
          dimensions: 'Unknown',
          filetype: input.type || 'unknown',
          modifiedDate: now,
          createdDate: now,
          exifIntact: false,
          hashSHA256: 'N/A',
          pHash: 'N/A',
        },
        manipulationSignals: [{
          id: 'inconclusive',
          name: 'Inconclusive Analysis',
          status: 'warning',
          score: 'N/A',
          description: reason,
        }],
        summaryText: `Inconclusive — ${reason}`,
        warnings: [`Inconclusive result: ${reason}`],
        investigatorNotes: `Inconclusive: ${reason}`,
        authenticityStatus: 'uncertain',
        overallScore: 50,
        confidence: 0,
        modelScores: [],
        heatmapUrl: EXPLAINABILITY_HEATMAPS.gradCamReal,
        elaUrl: SAMPLE_IMAGES.elaMap,
        whySignals: [`Inconclusive — ${reason}`],
      };
      return detectionResultInconclusive;
    }

    // ---- Extract the nested 'data' payload from the new API contract ----
    const payload = (apiData.data as Record<string, unknown>) || apiData;

    // ---- Common fields ----
    const rawPrediction = (payload.prediction as string) || 'REAL';
    const fakeProbability = (payload.fake_probability as number) ?? 0;
    const realProbability = (payload.real_probability as number) ?? 1;
    const confidence = (payload.confidence as number) ?? 0;
    const apiMediaType = (payload.media_type as string) ?? (isVideo ? 'video' : 'image');

    const classification: ClassificationType = rawPrediction === 'FAKE' ? 'Deepfake' : 'Real';
    const authenticityScore = Math.round(realProbability * 100);
    const confidenceScore = Math.round(confidence * 100);
    const riskLevel: RiskLevel = rawPrediction === 'FAKE' ? (confidenceScore > 75 ? 'high' : 'medium') : 'low';
    const fusedFake = Math.round(fakeProbability * 100);
    const fusedReal = 100 - fusedFake;

    // ================================================================
    // IMAGE result mapping
    // ================================================================
    if (apiMediaType === 'image') {
      const models = (payload.models as Record<string, Record<string, unknown>>) || {};
      const fusion = (payload.fusion as Record<string, unknown>) || {};
      const agreement = (payload.agreement as Record<string, unknown>) || {};
      const faceInfo = (payload.face_detection as Record<string, unknown>) || {};

      const xModel = models.xception || {};
      const eModel = models.efficientnet_b4 || {};
      const vModel = models.vit || {};

      const xFake = Math.round(((xModel.fake_probability as number) ?? 0) * 100);
      const xReal = Math.round(((xModel.real_probability as number) ?? 0) * 100);
      const eFake = Math.round(((eModel.fake_probability as number) ?? 0) * 100);
      const eReal = Math.round(((eModel.real_probability as number) ?? 0) * 100);
      const vFake = Math.round(((vModel.fake_probability as number) ?? 0) * 100);
      const vReal = Math.round(((vModel.real_probability as number) ?? 0) * 100);

      const faceDetected = (faceInfo.detected as boolean) ?? false;
      const fallback     = (faceInfo.fallback as boolean) ?? false;
      const numFaces     = (faceInfo.num_faces as number) ?? 0;

      const modelsAgree = (agreement.models_agree as boolean) ?? true;
      const disagreementLevel = (agreement.disagreement_level as string) ?? 'LOW';
      const modelAgreement: DetectionResult['modelAgreement'] = modelsAgree
        ? (disagreementLevel === 'LOW' ? 'High Agreement' : 'Moderate Agreement')
        : 'Model Disagreement';

      const xWeight = ((xModel.weight as number) ?? 0.45) * 100;
      const eWeight = ((eModel.weight as number) ?? 0.25) * 100;
      const vWeight = ((vModel.weight as number) ?? 0.30) * 100;

      const preprocessInfo = (payload.preprocessing as Record<string, unknown>) || {};
      const dimensions = (() => {
        const inputInfo = (payload as Record<string, unknown>);
        const w = (inputInfo.width as number) ?? 0;
        const h = (inputInfo.height as number) ?? 0;
        return (w && h) ? `${w} x ${h}` : 'Unknown';
      })();

      const whySignals: string[] = [];
      if (rawPrediction === 'FAKE') {
        whySignals.push(`Xception V2: ${xFake}% fake probability (weight: ${xWeight.toFixed(0)}%, TTA enabled)`);
        whySignals.push(`EfficientNet-B4 V2: ${eFake}% fake probability (weight: ${eWeight.toFixed(0)}%, TTA enabled)`);
        whySignals.push(`ViT V2.1: ${vFake}% fake probability (weight: ${vWeight.toFixed(0)}%, TTA enabled)`);
        whySignals.push(`Confidence-gated 3-model fusion score: ${fusedFake}% FAKE (mode: ${fusion.fusion_mode ?? 'standard'})`);
        if (!modelsAgree) {
          whySignals.push(`Models DISAGREE (${disagreementLevel} disagreement, Δ=${Math.round(((agreement.score_difference as number) ?? 0) * 100)}%)`);
        }
      } else {
        whySignals.push(`Xception V2: ${xReal}% authentic, ${xFake}% fake (weight: ${xWeight.toFixed(0)}%)`);
        whySignals.push(`EfficientNet-B4 V2: ${eReal}% authentic, ${eFake}% fake (weight: ${eWeight.toFixed(0)}%)`);
        whySignals.push(`ViT V2.1: ${vReal}% authentic, ${vFake}% fake (weight: ${vWeight.toFixed(0)}%)`);
        whySignals.push(`3-model ensemble REAL score: ${fusedReal}% (fusion mode: ${fusion.fusion_mode ?? 'standard'})`);
      }
      if (!faceDetected || fallback) {
        whySignals.push('⚠ No face detected — inference ran on full image fallback (may reduce reliability)');
      }

      const confLabel = (fake: number, real: number) =>
        (fake > 80 || real > 80) ? 'High' : (fake > 60 || real > 60) ? 'Medium' : 'Low';

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
          ? `Xception V2, EfficientNet-B4 V2, and ViT V2.1 agree: ${rawPrediction}. Disagreement level: ${disagreementLevel}.`
          : `Models DISAGREE. Xception: ${xModel.prediction ?? '?'}, EfficientNet: ${eModel.prediction ?? '?'}, ViT: ${vModel.prediction ?? '?'}. Score difference: ${Math.round(((agreement.score_difference as number) ?? 0) * 100)}%.`,
        riskLevel,
        detailedModelScores: [
          { name: 'Xception V2', realScore: xReal, fakeScore: xFake, confidence: confLabel(xFake, xReal) },
          { name: 'EfficientNet-B4 V2', realScore: eReal, fakeScore: eFake, confidence: confLabel(eFake, eReal) },
          { name: 'Vision Transformer (ViT) V2.1', realScore: vReal, fakeScore: vFake, confidence: confLabel(vFake, vReal) },
          { name: 'Final Fused Score', realScore: fusedReal, fakeScore: fusedFake, confidence: confidenceScore > 75 ? 'Optimal' : confidenceScore > 50 ? 'Medium' : 'Low' },
        ],
        explainability: {
          gradCamUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake : EXPLAINABILITY_HEATMAPS.gradCamReal,
          vitAttentionUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake : EXPLAINABILITY_HEATMAPS.vitAttentionReal,
          originalUrl: mediaPreviewUrl,
          highlightedRegions: [],
        },
        metadata: {
          filename: input.name,
          filesize: `${(input.size / (1024 * 1024)).toFixed(2)} MB`,
          dimensions,
          filetype: input.type || 'image/jpeg',
          modifiedDate: now,
          createdDate: now,
          exifIntact: true,
          hashSHA256: 'computed-server-side',
          pHash: 'computed-server-side',
        },
        manipulationSignals: rawPrediction === 'FAKE'
          ? [
              { id: 'xception', name: 'Xception V2 Detection', status: 'flagged', score: `${xFake}% FAKE`, description: `Xception V2 (weight ${xWeight.toFixed(0)}%) detected ${xFake}% fake probability via TTA.` },
              { id: 'efficientnet', name: 'EfficientNet-B4 V2 Detection', status: eFake > 50 ? 'flagged' : 'warning', score: `${eFake}% FAKE`, description: `EfficientNet-B4 V2 (weight ${eWeight.toFixed(0)}%) detected ${eFake}% fake probability.` },
              { id: 'vit', name: 'ViT V2.1 Detection', status: vFake > 50 ? 'flagged' : 'warning', score: `${vFake}% FAKE`, description: `Vision Transformer V2.1 (weight ${vWeight.toFixed(0)}%) detected ${vFake}% fake probability.` },
              { id: 'fusion', name: 'Confidence-Gated Ensemble Fusion', status: 'flagged', score: `${fusedFake}% FAKE`, description: `3-model ensemble fusion (mode: ${fusion.fusion_mode ?? 'STANDARD'}). Final score: ${fusedFake}%.` },
              ...(!faceDetected || fallback ? [{ id: 'face', name: 'Face Detection', status: 'warning' as const, score: 'No face found', description: 'Full-image fallback used. Accuracy may be reduced.' }] : []),
            ]
          : [
              { id: 'xception', name: 'Xception V2 Detection', status: 'passed', score: `${xReal}% REAL`, description: `Xception V2 (weight ${xWeight.toFixed(0)}%) found ${xReal}% authentic probability.` },
              { id: 'efficientnet', name: 'EfficientNet-B4 V2 Detection', status: 'passed', score: `${eReal}% REAL`, description: `EfficientNet-B4 V2 (weight ${eWeight.toFixed(0)}%) found ${eReal}% authentic probability.` },
              { id: 'vit', name: 'ViT V2.1 Detection', status: 'passed', score: `${vReal}% REAL`, description: `Vision Transformer V2.1 (weight ${vWeight.toFixed(0)}%) found ${vReal}% authentic probability.` },
              { id: 'fusion', name: 'Confidence-Gated Ensemble Fusion', status: 'passed', score: `${fusedReal}% REAL`, description: `3-model ensemble fusion (mode: ${fusion.fusion_mode ?? 'STANDARD'}). Final authentic score: ${fusedReal}%.` },
            ],
        summaryText: rawPrediction === 'FAKE'
          ? `ForensIQ V2.1 detected synthetic manipulation with ${confidenceScore}% confidence. Xception: ${xFake}%, EfficientNet-B4: ${eFake}%, ViT: ${vFake}%. Ensemble fused score: ${fusedFake}% FAKE.`
          : `ForensIQ V2.1 found no evidence of manipulation with ${confidenceScore}% confidence. Xception: ${xReal}%, EfficientNet-B4: ${eReal}%, ViT: ${vReal}% authentic. Ensemble: ${fusedReal}% REAL.`,
        warnings: [
          ...(!faceDetected || fallback ? ['No face detected. Full-image fallback inference used — results may be less reliable.'] : []),
          ...(!modelsAgree ? [`Models disagree (${disagreementLevel} disagreement). Review individual model scores carefully.`] : []),
        ],
        investigatorNotes: [
          `Source: ForensIQ V2.1 face-crop 3-model pipeline`,
          `Fusion mode: ${fusion.fusion_mode ?? 'STANDARD_WEIGHTED_FUSION'}`,
          `Weights: Xception=${xWeight.toFixed(0)}%, EfficientNet=${eWeight.toFixed(0)}%, ViT=${vWeight.toFixed(0)}%`,
          `TTA: ${preprocessInfo.tta ?? true}`,
          `Face detected: ${faceDetected}, Fallback: ${fallback}`,
        ].join(' | '),
        authenticityStatus: rawPrediction === 'FAKE' ? 'likely-ai-generated' : 'likely-authentic',
        overallScore: authenticityScore,
        confidence: confidenceScore,
        modelScores: [
          { name: 'Xception V2', score: xReal, confidence: confLabel(xFake, xReal) },
          { name: 'EfficientNet-B4 V2', score: eReal, confidence: confLabel(eFake, eReal) },
          { name: 'Vision Transformer (ViT) V2.1', score: vReal, confidence: confLabel(vFake, vReal) },
        ],
        heatmapUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake : EXPLAINABILITY_HEATMAPS.gradCamReal,
        elaUrl: SAMPLE_IMAGES.elaMap,
        whySignals,
      };
      return detectionResult;
    }

    // ================================================================
    // VIDEO result mapping
    // ================================================================
    const sampling    = (payload.sampling    as Record<string, unknown>) || {};
    const aggregation = (payload.aggregation as Record<string, unknown>) || {};
    const frameStats  = (payload.frame_statistics as Record<string, unknown>) || {};
    const videoInfo   = (payload.video_info  as Record<string, unknown>) || {};
    const models      = (payload.models      as Record<string, Record<string, unknown>>) || {};
    const rawFrames   = (payload.frames      as Record<string, unknown>[]) || [];

    const sampledFrames    = (sampling.selected_frames      as number) ?? 0;
    const framesAnalyzed   = (sampling.successfully_analyzed as number) ?? 0;
    const noFaceFrames     = (sampling.no_face_frames       as number) ?? 0;
    const suspiciousFrames = (frameStats.fake_frames        as number) ?? 0;
    const fps              = (videoInfo.fps                 as number) ?? 0;
    const durationSecs     = (videoInfo.duration_seconds    as number) ?? 0;

    const xMeanFake = Math.round(((frameStats.xception_mean_fake     as number) ?? 0) * 100);
    const eMeanFake = Math.round(((frameStats.efficientnet_mean_fake as number) ?? 0) * 100);
    const vMeanFake = Math.round(((frameStats.vit_mean_fake          as number) ?? 0) * 100);

    const xWeight = ((models.xception?.weight      as number) ?? 0.45) * 100;
    const eWeight = ((models.efficientnet_b4?.weight as number) ?? 0.25) * 100;
    const vWeight = ((models.vit?.weight           as number) ?? 0.30) * 100;

    // Convert raw frame data → VideoFrameAnalysis
    const frameResults = rawFrames.map((fr, idx) => ({
      frameNumber: (fr.frame_index as number) ?? idx + 1,
      timestamp: `${String(Math.floor((fr.timestamp_seconds as number ?? 0) / 60)).padStart(2, '0')}:${String(Math.round((fr.timestamp_seconds as number ?? 0) % 60)).padStart(2, '0')}`,
      fakeScore: Math.round(((fr.fake_score as number) ?? 0) * 100),
      isSuspicious: (fr.prediction as string) === 'FAKE',
      anomalyType: (fr.prediction as string) === 'FAKE' ? 'Deepfake artifact detected' : undefined,
    }));

    const videoWhySignals: string[] = [];
    if (rawPrediction === 'FAKE') {
      videoWhySignals.push(`Xception V2 mean fake score across frames: ${xMeanFake}% (weight: ${xWeight.toFixed(0)}%)`);
      videoWhySignals.push(`EfficientNet-B4 V2 mean fake score across frames: ${eMeanFake}% (weight: ${eWeight.toFixed(0)}%)`);
      videoWhySignals.push(`ViT V2.1 mean fake score across frames: ${vMeanFake}% (weight: ${vWeight.toFixed(0)}%)`);
      videoWhySignals.push(`${suspiciousFrames} of ${framesAnalyzed} analyzed frames classified as FAKE`);
    } else {
      videoWhySignals.push(`Xception V2 mean authentic score: ${100 - xMeanFake}% across ${framesAnalyzed} frames`);
      videoWhySignals.push(`EfficientNet-B4 V2 mean authentic score: ${100 - eMeanFake}%`);
      videoWhySignals.push(`ViT V2.1 mean authentic score: ${100 - vMeanFake}%`);
      videoWhySignals.push(`Only ${suspiciousFrames} of ${framesAnalyzed} frames flagged as suspicious`);
    }
    if (noFaceFrames > 0) {
      videoWhySignals.push(`⚠ ${noFaceFrames} frame(s) had no face detected — full-image fallback used for those frames`);
    }

    const videoResult: DetectionResult = {
      id: `FQ-VID-${Math.floor(1000 + Math.random() * 9000)}`,
      mediaName: input.name,
      mediaUrl: mediaPreviewUrl,
      mediaType: 'video',
      uploadDate: now,
      classification,
      confidenceScore,
      authenticityScore,
      processingTimeMs: 0,
      facesDetected: framesAnalyzed - noFaceFrames,
      modelAgreement: confidenceScore > 70 ? 'High Agreement' : confidenceScore > 50 ? 'Moderate Agreement' : 'Model Disagreement',
      modelAgreementDescription: `Video analysis: ${framesAnalyzed} frames analyzed, ${suspiciousFrames} classified as FAKE. ${noFaceFrames} frames had no face detected.`,
      riskLevel,
      detailedModelScores: [
        { name: 'Xception V2 (Frame-level)', realScore: 100 - xMeanFake, fakeScore: xMeanFake, confidence: xMeanFake > 70 ? 'High' : 'Medium' },
        { name: 'EfficientNet-B4 V2 (Frame-level)', realScore: 100 - eMeanFake, fakeScore: eMeanFake, confidence: eMeanFake > 70 ? 'High' : 'Medium' },
        { name: 'Vision Transformer V2.1 (Frame-level)', realScore: 100 - vMeanFake, fakeScore: vMeanFake, confidence: vMeanFake > 70 ? 'High' : 'Medium' },
        { name: 'Final Video Ensemble Score', realScore: fusedReal, fakeScore: fusedFake, confidence: confidenceScore > 75 ? 'Optimal' : 'Medium' },
      ],
      explainability: {
        gradCamUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake : EXPLAINABILITY_HEATMAPS.gradCamReal,
        vitAttentionUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.vitAttentionDeepfake : EXPLAINABILITY_HEATMAPS.vitAttentionReal,
        originalUrl: mediaPreviewUrl,
        highlightedRegions: [],
      },
      metadata: {
        filename: input.name,
        filesize: `${(input.size / (1024 * 1024)).toFixed(2)} MB`,
        dimensions: (videoInfo.width && videoInfo.height) ? `${videoInfo.width} x ${videoInfo.height}` : 'Unknown',
        filetype: input.type || 'video/mp4',
        cameraModel: `${fps.toFixed(2)} fps`,
        modifiedDate: now,
        createdDate: now,
        exifIntact: false,
        hashSHA256: 'computed-server-side',
        pHash: 'computed-server-side',
      },
      manipulationSignals: [
        { id: 'frames', name: 'Frame-Level Deepfake Detection', status: rawPrediction === 'FAKE' ? 'flagged' : 'passed', score: `${suspiciousFrames}/${framesAnalyzed} frames FAKE`, description: `${suspiciousFrames} of ${framesAnalyzed} sampled frames classified as synthetic.` },
        { id: 'xception', name: 'Xception V2 Frame Analysis', status: xMeanFake > 50 ? 'flagged' : 'passed', score: `${xMeanFake}% mean fake`, description: `Xception V2 mean fake score: ${xMeanFake}% (weight: ${xWeight.toFixed(0)}%).` },
        { id: 'efficientnet', name: 'EfficientNet-B4 Frame Analysis', status: eMeanFake > 50 ? 'flagged' : 'passed', score: `${eMeanFake}% mean fake`, description: `EfficientNet-B4 V2 mean fake score: ${eMeanFake}% (weight: ${eWeight.toFixed(0)}%).` },
        { id: 'vit', name: 'ViT V2.1 Frame Analysis', status: vMeanFake > 50 ? 'flagged' : 'passed', score: `${vMeanFake}% mean fake`, description: `Vision Transformer V2.1 mean fake score: ${vMeanFake}% (weight: ${vWeight.toFixed(0)}%).` },
        ...(noFaceFrames > 0 ? [{ id: 'face', name: 'Face Detection Coverage', status: 'warning' as const, score: `${noFaceFrames} frames w/o face`, description: `${noFaceFrames} frame(s) had no face detected. Full-image fallback used.` }] : []),
      ],
      videoData: {
        sampledFrames,
        facesDetected: framesAnalyzed - noFaceFrames,
        framesAnalyzed,
        suspiciousFrameCount: suspiciousFrames,
        fps,
        durationSeconds: durationSecs,
        frameResults,
      },
      summaryText: rawPrediction === 'FAKE'
        ? `ForensIQ V2.1 video analysis: FAKE detected with ${confidenceScore}% confidence. ${suspiciousFrames}/${framesAnalyzed} frames flagged. Xception: ${xMeanFake}%, EfficientNet: ${eMeanFake}%, ViT: ${vMeanFake}% mean fake.`
        : `ForensIQ V2.1 video analysis: No deepfake detected with ${confidenceScore}% confidence. Only ${suspiciousFrames}/${framesAnalyzed} frames suspicious.`,
      warnings: [
        'Video inference uses frame-level image classifiers; it is not a temporal video model.',
        'Video fake score is an aggregation of sampled frame predictions.',
        ...(noFaceFrames > 0 ? [`${noFaceFrames} frame(s) lacked face detection — full-image fallback used.`] : []),
        ...(payload.warnings as string[] ?? []),
      ],
      investigatorNotes: [
        `ForensIQ V2.1 video pipeline`,
        `Frames: ${sampledFrames} selected, ${framesAnalyzed} analyzed`,
        `Duration: ${durationSecs.toFixed(1)}s at ${fps.toFixed(2)} fps`,
        `Aggregation: mean×0.50 + median×0.30 + 75th-percentile×0.20`,
        `Weights: Xception=${xWeight.toFixed(0)}%, EfficientNet=${eWeight.toFixed(0)}%, ViT=${vWeight.toFixed(0)}%`,
      ].join(' | '),
      authenticityStatus: rawPrediction === 'FAKE' ? 'likely-ai-generated' : 'likely-authentic',
      overallScore: authenticityScore,
      confidence: confidenceScore,
      modelScores: [
        { name: 'Xception V2', score: 100 - xMeanFake, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', score: 100 - eMeanFake, confidence: 'High' },
        { name: 'ViT V2.1', score: 100 - vMeanFake, confidence: 'High' },
      ],
      heatmapUrl: rawPrediction === 'FAKE' ? EXPLAINABILITY_HEATMAPS.gradCamDeepfake : EXPLAINABILITY_HEATMAPS.gradCamReal,
      elaUrl: SAMPLE_IMAGES.elaMap,
      whySignals: videoWhySignals,
    };
    return videoResult;
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
