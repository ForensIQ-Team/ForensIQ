import { DetectionResult, HistoryItem } from '../types';

const STORAGE_KEY = 'forensiq_analysis_history';

// Default seed history matching exact TypeScript types
const DEFAULT_SEED_HISTORY: HistoryItem[] = [
  {
    id: 'FQ-2026-9042',
    mediaName: 'Portrait_Authentic_01.jpg',
    mediaUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
    action: 'Check Media',
    resultSummary: 'REAL (99.8% Confidence)',
    date: '2026-09-02 18:42 UTC',
    status: 'Completed',
    risk: 'low',
    detectionResult: {
      id: 'FQ-2026-9042',
      mediaName: 'Portrait_Authentic_01.jpg',
      mediaUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
      mediaType: 'image',
      uploadDate: '2026-09-02 18:42 UTC',
      classification: 'Real',
      confidenceScore: 99.8,
      authenticityScore: 99.8,
      processingTimeMs: 412,
      facesDetected: 1,
      modelAgreement: 'High Agreement',
      modelAgreementDescription: 'Both Xception V2 and EfficientNet-B4 V2 concur media is authentic.',
      riskLevel: 'low',
      detailedModelScores: [
        { name: 'Xception V2', realScore: 99.7, fakeScore: 0.3, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', realScore: 100.0, fakeScore: 0.0, confidence: 'Optimal' },
        { name: 'ForensIQ Score Fusion', realScore: 99.8, fakeScore: 0.2, confidence: 'High' }
      ],
      explainability: {
        gradCamUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
        vitAttentionUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
        originalUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
        highlightedRegions: [
          { label: 'Facial Boundary', description: 'Natural skin transition', bounds: { x: 20, y: 20, width: 60, height: 60 } }
        ]
      },
      metadata: {
        filename: 'Portrait_Authentic_01.jpg',
        filetype: 'image/jpeg',
        filesize: '2.4 MB',
        dimensions: '1920x1080',
        cameraModel: 'Sony ILCE-7M4',
        modifiedDate: '2026-09-02 18:40 UTC',
        createdDate: '2026-09-02 18:40 UTC',
        exifIntact: true,
        hashSHA256: '0xa498f12c8b9104ef7120489912bc091d',
        pHash: 'phash-auth-9042'
      },
      manipulationSignals: [
        { id: '1', name: 'Facial Boundary Alignment', status: 'passed', description: 'No boundary splicing detected', score: '99.8% PASS' },
        { id: '2', name: 'Color Space & Texture', status: 'passed', description: 'Natural sensor noise pattern', score: '99.5% PASS' }
      ],
      summaryText: 'Forensic evaluation indicates high probability of authentic camera capture with no synthetic face-swap artifacts.',
      warnings: [],
      authenticityStatus: 'likely-authentic',
      overallScore: 99.8,
      confidence: 99.8,
      modelScores: [
        { name: 'Xception V2', score: 99.7, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', score: 100.0, confidence: 'Optimal' }
      ],
      whySignals: ['Natural skin grain and pore-level texture', 'Consistent camera sensor noise profile']
    }
  },
  {
    id: 'FQ-2026-4192',
    mediaName: 'suspect_face_swap.png',
    mediaUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80',
    action: 'Check Media',
    resultSummary: 'DEEPFAKE (96.4% Fake)',
    date: '2026-09-02 17:15 UTC',
    status: 'Flagged',
    risk: 'high',
    detectionResult: {
      id: 'FQ-2026-4192',
      mediaName: 'suspect_face_swap.png',
      mediaUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80',
      mediaType: 'image',
      uploadDate: '2026-09-02 17:15 UTC',
      classification: 'Deepfake',
      confidenceScore: 96.4,
      authenticityScore: 3.6,
      processingTimeMs: 489,
      facesDetected: 1,
      modelAgreement: 'High Agreement',
      modelAgreementDescription: 'Both models confirm synthetic face swap boundary anomalies.',
      riskLevel: 'high',
      detailedModelScores: [
        { name: 'Xception V2', realScore: 2.9, fakeScore: 97.1, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', realScore: 5.2, fakeScore: 94.8, confidence: 'High' },
        { name: 'ForensIQ Score Fusion', realScore: 3.6, fakeScore: 96.4, confidence: 'High' }
      ],
      explainability: {
        gradCamUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80',
        vitAttentionUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80',
        originalUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80',
        highlightedRegions: [
          { label: 'Synthetic Jawline', description: 'Blending boundary anomaly', bounds: { x: 15, y: 40, width: 70, height: 40 } }
        ]
      },
      metadata: {
        filename: 'suspect_face_swap.png',
        filetype: 'image/png',
        filesize: '1.8 MB',
        dimensions: '1080x1080',
        modifiedDate: '2026-09-02 17:10 UTC',
        createdDate: '2026-09-02 17:10 UTC',
        exifIntact: false,
        hashSHA256: '0x8f190c7b2a4192d19e55a0192f10b74d',
        pHash: 'phash-fake-4192'
      },
      manipulationSignals: [
        { id: '1', name: 'Facial Boundary Alignment', status: 'flagged', description: 'Blending artifacts around chin line', score: '97.1% FAKE' },
        { id: '2', name: 'Sensor Noise Discontinuity', status: 'flagged', description: 'Disruptive frequency spectrum', score: '94.8% FAKE' }
      ],
      summaryText: 'High confidence synthetic facial replacement detected. Neural models flag irregular boundary blending.',
      warnings: ['Synthetic face-swap detected', 'High risk score'],
      authenticityStatus: 'likely-ai-generated',
      overallScore: 3.6,
      confidence: 96.4,
      modelScores: [
        { name: 'Xception V2', score: 97.1, confidence: 'High' },
        { name: 'EfficientNet-B4 V2', score: 94.8, confidence: 'High' }
      ],
      whySignals: ['Blending boundary around face oval', 'Unnatural frequency dropoff in eye region']
    }
  }
];

export const historyService = {
  getHistoryItems(): HistoryItem[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          return parsed;
        }
      }
    } catch (e) {
      console.warn('Failed to parse history from localStorage', e);
    }
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_SEED_HISTORY));
    } catch (e) {
      // ignore
    }
    return DEFAULT_SEED_HISTORY;
  },

  addHistoryItem(result: DetectionResult): HistoryItem {
    const history = this.getHistoryItems();
    
    // Generate unique ID in FQ-2026-XXXX format
    const existingIds = new Set(history.map(h => h.id));
    let randomId = `FQ-2026-${Math.floor(1000 + Math.random() * 9000)}`;
    while (existingIds.has(randomId)) {
      randomId = `FQ-2026-${Math.floor(1000 + Math.random() * 9000)}`;
    }

    const isFake = result.classification === 'Deepfake' || result.classification === 'AI-Generated';
    const confValue = result.confidenceScore ?? result.confidence ?? 90;
    const confidence = confValue.toFixed(1);
    
    const updatedResult: DetectionResult = {
      ...result,
      id: randomId,
    };

    const newItem: HistoryItem = {
      id: randomId,
      mediaName: result.mediaName,
      mediaUrl: result.mediaUrl || 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=300&q=80',
      action: 'Check Media',
      resultSummary: `${isFake ? 'DEEPFAKE' : 'REAL'} (${confidence}% Confidence)`,
      date: result.uploadDate || new Date().toISOString().replace('T', ' ').substring(0, 16) + ' UTC',
      status: isFake ? 'Flagged' : 'Completed',
      risk: result.riskLevel || (isFake ? 'high' : 'low'),
      detectionResult: updatedResult,
    };

    const updatedHistory = [newItem, ...history.filter(h => h.id !== randomId)];
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory));
    } catch (e) {
      console.warn('Could not save history item to localStorage', e);
    }

    return newItem;
  },

  getAnalysisById(id: string): DetectionResult | undefined {
    const history = this.getHistoryItems();
    const found = history.find(h => h.id === id || h.id.replace('HIST-', 'FQ-2026-') === id);
    if (found?.detectionResult) {
      return found.detectionResult;
    }
    return undefined;
  }
};
