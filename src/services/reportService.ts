export interface GenerateReportResponse {
  status: 'success';
  report_id: string;
  report_url: string;
  download_url: string;
  verdict: string;
  confidence: number;
}

const API_BASE_URL = 'http://localhost:8000';

/**
 * Upload an image or video file to generate a Groq-powered ForensIQ Forensic Report.
 */
export async function generateForensicReport(file: File): Promise<GenerateReportResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/report`, {
    method: 'POST',
    body: formData,
    // Note: Do NOT set Content-Type header — browser automatically sets multipart/form-data with boundary
  });

  if (!response.ok) {
    let errText = '';
    try {
      const errJson = await response.json();
      errText = errJson.detail || '';
    } catch {
      errText = await response.text().catch(() => '');
    }
    throw new Error(
      `Report generation failed (${response.status}): ${errText || response.statusText}`
    );
  }

  return response.json();
}

export function buildReportHtmlUrl(reportId: string): string {
  return `${API_BASE_URL}/api/v1/report/${encodeURIComponent(reportId)}/html`;
}

export function buildReportDownloadUrl(reportId: string): string {
  return `${API_BASE_URL}/api/v1/report/${encodeURIComponent(reportId)}/download`;
}

export const reportService = {
  generateReport: generateForensicReport,
  getReportHtmlUrl: buildReportHtmlUrl,
  getReportDownloadUrl: buildReportDownloadUrl,
};
