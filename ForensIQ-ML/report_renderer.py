"""
ForensIQ Report Renderer
========================
Renders a report dict (from ForensicReportGenerator) as a fully
self-contained HTML file with base64-encoded images.
"""

import base64
import json
import shutil
from pathlib import Path
from datetime import datetime


def _b64_image(path: Path) -> str:
    """Return a base64 data-URI for the image at path, or empty string."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.suffix.lower().lstrip(".")
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                "webp": "webp", "bmp": "bmp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


def render_html(report: dict, output_dir: Path) -> Path:
    """
    Render report.html into output_dir.

    Images are read from output_dir/assets/ (which report_generator.py
    already populated) and embedded as base64 into the HTML so the file
    is fully self-contained.

    Returns the path to the written report.html.
    """
    output_dir = Path(output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Pull data from the report dict
    # ------------------------------------------------------------------
    report_id    = report.get("report_id", "—")
    generated_at = report.get("generated_at", "")
    version      = report.get("version", "ForensIQ-Report-V1")

    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        date_str = dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        date_str = generated_at

    inp     = report.get("input", {})
    verdict = report.get("verdict", {})
    visuals = report.get("visuals", {})
    ai      = report.get("ai_analysis", {})
    ai_err  = report.get("ai_analysis_error")
    meta    = report.get("metadata", {})
    vstat   = report.get("video_stats")

    label      = str(verdict.get("label", "UNKNOWN")).upper()
    confidence = float(verdict.get("confidence", 0.0))
    fake_score = float(verdict.get("fake_score", 0.0))
    headline   = str(verdict.get("headline", ""))
    summary    = str(ai.get("summary", ""))
    analysis   = ai.get("analysis", {})
    indicators = ai.get("observed_indicators", [])
    conf_stmt  = str(ai.get("confidence_statement", ""))
    limitations = str(ai.get("limitations", ""))
    recommendation = str(ai.get("recommendation", ""))

    media_type = str(inp.get("media_type", "image"))
    filename   = str(inp.get("filename", ""))
    img_w      = int(inp.get("width", 0))
    img_h      = int(inp.get("height", 0))

    # ------------------------------------------------------------------
    # Images (base64)
    # Derive the original-image filename from the report dict so we
    # handle .png / .jpg / .webp uploads correctly.
    # ------------------------------------------------------------------
    _orig_rel  = visuals.get("original_image", "assets/original.jpg")
    _orig_name = Path(_orig_rel).name                 # e.g. "original.jpg" or "original.png"
    _orig_path = assets_dir / _orig_name

    # Fallback: if the expected file is missing, look for any image in assets/
    if not _orig_path.exists():
        _img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        _candidates = [
            p for p in assets_dir.iterdir()
            if p.suffix.lower() in _img_exts and "overlay" not in p.name.lower()
        ]
        if _candidates:
            _orig_path = _candidates[0]
            print(f"[ReportRenderer][warn] Expected {_orig_name} not found; "
                  f"using {_orig_path.name} instead.")
        else:
            print(f"[ReportRenderer][warn] No original image found in {assets_dir}")

    orig_b64    = _b64_image(_orig_path)
    overlay_b64 = _b64_image(assets_dir / "gradcam_overlay.jpg")

    # Sanity log
    _orig_sz    = _orig_path.stat().st_size if _orig_path.exists() else 0
    _overlay_sz = (assets_dir / "gradcam_overlay.jpg").stat().st_size if (assets_dir / "gradcam_overlay.jpg").exists() else 0
    print(f"[ReportRenderer] original image size : {_orig_sz:,} bytes ({_orig_path.name})")
    print(f"[ReportRenderer] overlay image size  : {_overlay_sz:,} bytes")
    if _orig_sz < 1000:
        print(f"[ReportRenderer][WARN] original image is suspiciously small ({_orig_sz} bytes)!")
    if _overlay_sz > 0 and _overlay_sz < 1000:
        print(f"[ReportRenderer][WARN] overlay image is suspiciously small ({_overlay_sz} bytes)!")

    # ------------------------------------------------------------------
    # Verdict colour
    # ------------------------------------------------------------------
    if label == "FAKE":
        badge_bg = "#dc2626"; badge_text = "#fff"
        bar_color = "#dc2626"
    elif label == "REAL":
        badge_bg = "#16a34a"; badge_text = "#fff"
        bar_color = "#16a34a"
    else:  # SUSPICIOUS / INCONCLUSIVE
        badge_bg = "#d97706"; badge_text = "#fff"
        bar_color = "#d97706"

    conf_pct = f"{confidence * 100:.1f}%"
    fake_pct = f"{fake_score * 100:.1f}%"

    # ------------------------------------------------------------------
    # Metadata table rows
    # ------------------------------------------------------------------
    bbox      = meta.get("bbox")
    bbox_str  = f"[{', '.join(str(v) for v in bbox)}]" if bbox else "N/A"
    bfrac     = meta.get("bbox_fraction")
    bfrac_str = f"{bfrac * 100:.2f}%" if bfrac is not None else "N/A"

    meta_rows = f"""
        <tr><td>File</td><td>{filename}</td></tr>
        <tr><td>Type</td><td>{media_type.capitalize()}</td></tr>
        <tr><td>Dimensions</td><td>{img_w} × {img_h} px</td></tr>
        <tr><td>Face detected</td><td>{"Yes" if meta.get("face_detected") else "No"}</td></tr>
        <tr><td>Detection method</td><td>{meta.get("detection_method","N/A")}</td></tr>
        <tr><td>Alignment applied</td><td>{"Yes" if meta.get("aligned") else "No"}</td></tr>
        <tr><td>Attention bbox [x,y,w,h]</td><td>{bbox_str}</td></tr>
        <tr><td>Bbox coverage</td><td>{bfrac_str}</td></tr>
        <tr><td>Fake score (raw)</td><td>{fake_pct}</td></tr>
        <tr><td>Report ID</td><td>{report_id}</td></tr>
        <tr><td>Generated</td><td>{date_str}</td></tr>
        <tr><td>Version</td><td>{version}</td></tr>
    """

    # Video-specific rows
    if vstat:
        frames_a = vstat.get("frames_analyzed", "N/A")
        fake_f   = vstat.get("fake_frames", "N/A")
        fratio   = vstat.get("fake_frame_ratio")
        fratio_s = f"{fratio*100:.1f}%" if fratio is not None else "N/A"
        p75      = vstat.get("p75_fake")
        p90      = vstat.get("p90_fake")
        p75_s    = f"{p75:.3f}" if isinstance(p75, (int, float)) else "N/A"
        p90_s    = f"{p90:.3f}" if isinstance(p90, (int, float)) else "N/A"
        peak_esc = vstat.get("peak_escalated", "N/A")
        seg_pat  = vstat.get("segment_pattern", "N/A")
        meta_rows += f"""
        <tr><td>Frames analyzed</td><td>{frames_a}</td></tr>
        <tr><td>Fake frames</td><td>{fake_f} ({fratio_s})</td></tr>
        <tr><td>Score p75 / p90</td><td>{p75_s} / {p90_s}</td></tr>
        <tr><td>Peak escalation</td><td>{peak_esc}</td></tr>
        <tr><td>Temporal pattern</td><td>{seg_pat}</td></tr>
        """

    # Indicators HTML
    if indicators:
        indicators_html = "\n".join(
            f'<li>{item}</li>' for item in indicators
        )
        indicators_block = f'<ul class="indicators">{indicators_html}</ul>'
    else:
        indicators_block = "<p><em>No indicators available.</em></p>"

    # AI error notice
    ai_error_notice = ""
    if ai_err:
        ai_error_notice = f"""
        <div class="notice warning">
          AI narrative analysis unavailable: {ai_err}
        </div>"""

    # Overlay notice
    overlay_notice = ""
    if not overlay_b64:
        gradcam_err = visuals.get("gradcam_error", "")
        overlay_notice = f"""
        <div class="notice warning">
          Grad-CAM overlay unavailable{': ' + gradcam_err if gradcam_err else ''}.
        </div>"""

    # ------------------------------------------------------------------
    # Build HTML
    # ------------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ForensIQ Report — {report_id}</title>
<meta name="description" content="ForensIQ forensic analysis report {report_id} — {label} verdict with {conf_pct} confidence.">
<style>
  /* ================================================================
     DESIGN SYSTEM
     ================================================================ */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: #1f2937;
    background: #f7f8fa;
  }}

  .wrap {{ max-width: 820px; margin: 0 auto; padding: 32px 20px 60px; }}

  /* ---- Banner ---- */
  .banner {{
    background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
    color: #fff;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    flex-wrap: wrap;
  }}
  .banner-logo {{ font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }}
  .banner-logo span {{ color: #60a5fa; }}
  .banner-meta {{ font-size: 12px; color: #9ca3af; text-align: right; line-height: 1.8; }}
  .banner-meta strong {{ color: #d1d5db; }}

  /* ---- Cards ---- */
  .card {{
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    padding: 28px 32px;
    margin-bottom: 20px;
  }}

  .section-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #6b7280;
    font-weight: 700;
    margin-bottom: 16px;
  }}

  /* ---- Verdict ---- */
  .verdict-row {{
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .verdict-badge {{
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 1px;
    padding: 6px 22px;
    border-radius: 8px;
    background: {badge_bg};
    color: {badge_text};
  }}
  .verdict-confidence {{
    font-size: 15px;
    color: #374151;
  }}
  .verdict-confidence strong {{ font-size: 22px; color: {badge_bg}; }}

  .confidence-bar-wrap {{
    background: #f3f4f6;
    border-radius: 999px;
    height: 8px;
    margin: 8px 0 16px;
    overflow: hidden;
  }}
  .confidence-bar {{
    height: 100%;
    border-radius: 999px;
    background: {bar_color};
    width: {conf_pct};
    transition: width 0.5s ease;
  }}

  .headline {{
    font-size: 18px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 10px;
  }}
  .summary {{ color: #374151; }}

  /* ---- Visual Evidence ---- */
  .image-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 8px;
  }}
  @media (max-width: 560px) {{ .image-grid {{ grid-template-columns: 1fr; }} }}
  .image-cell {{ text-align: center; }}
  .image-cell img {{
    width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    display: block;
  }}
  .image-label {{
    font-size: 12px;
    color: #6b7280;
    margin-top: 6px;
  }}
  .img-caption {{
    font-size: 12px;
    color: #9ca3af;
    margin-top: 10px;
    font-style: italic;
  }}

  /* ---- Analysis ---- */
  .analysis-section {{ margin-bottom: 18px; }}
  .analysis-section h4 {{
    font-size: 13px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
  }}
  .analysis-section p {{ color: #4b5563; }}

  /* ---- Indicators ---- */
  ul.indicators {{
    padding-left: 20px;
    color: #374151;
  }}
  ul.indicators li {{ margin-bottom: 6px; }}

  /* ---- Metadata table ---- */
  .meta-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .meta-table td {{
    padding: 7px 10px;
    border-bottom: 1px solid #f3f4f6;
    vertical-align: top;
  }}
  .meta-table td:first-child {{
    color: #6b7280;
    font-weight: 600;
    white-space: nowrap;
    width: 40%;
  }}

  /* ---- Notice ---- */
  .notice {{
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    margin-top: 12px;
  }}
  .notice.warning {{ background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }}

  /* ---- Footer ---- */
  .footer {{
    text-align: center;
    font-size: 12px;
    color: #9ca3af;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #e5e7eb;
  }}

  /* ---- Print ---- */
  @media print {{
    body {{ background: #fff; }}
    .card {{ box-shadow: none; page-break-inside: avoid; }}
    .banner {{ background: #1f2937 !important; -webkit-print-color-adjust: exact; }}
  }}
</style>
</head>
<body>
<div class="wrap">

  <!-- BANNER -->
  <div class="banner">
    <div>
      <div class="banner-logo">Forens<span>IQ</span></div>
      <div style="font-size:12px;color:#9ca3af;margin-top:4px;">
        AI-Powered Digital Forensics Report
      </div>
    </div>
    <div class="banner-meta">
      <strong>{report_id}</strong><br>
      {date_str}<br>
      {version}
    </div>
  </div>

  <!-- 1. VERDICT -->
  <div class="card">
    <div class="section-label">1 · Verdict</div>
    <div class="verdict-row">
      <div class="verdict-badge">{label}</div>
      <div class="verdict-confidence">
        Confidence&nbsp;<strong>{conf_pct}</strong>
      </div>
    </div>
    <div class="confidence-bar-wrap">
      <div class="confidence-bar"></div>
    </div>
    {'<div class="headline">' + headline + '</div>' if headline else ''}
    {'<p class="summary">' + summary + '</p>' if summary else ''}
    {ai_error_notice}
  </div>

  <!-- 2. VISUAL EVIDENCE -->
  <div class="card">
    <div class="section-label">2 · Visual Evidence</div>
    <div class="image-grid">
      <div class="image-cell">
        {'<img src="' + orig_b64 + '" alt="Original image">' if orig_b64 else '<p style="color:#9ca3af;font-size:13px;padding:40px 0;">Original image unavailable</p>'}
        <div class="image-label">Original</div>
      </div>
      <div class="image-cell">
        {'<img src="' + overlay_b64 + '" alt="Grad-CAM attention overlay">' if overlay_b64 else '<p style="color:#9ca3af;font-size:13px;padding:40px 0;">Attention map unavailable</p>'}
        <div class="image-label">Attention Map Overlay</div>
      </div>
    </div>
    <p class="img-caption">
      Attention map showing regions the analysis pipeline focused on.
      Warm colours (yellow/red) indicate higher model attention.
    </p>
    {overlay_notice}
  </div>

  <!-- 3. AI ANALYSIS -->
  <div class="card">
    <div class="section-label">3 · AI Analysis</div>
    {''.join([
      f'<div class="analysis-section"><h4>Overview</h4><p>{analysis.get("overview","")}</p></div>',
      f'<div class="analysis-section"><h4>Heatmap Interpretation</h4><p>{analysis.get("heatmap_interpretation","")}</p></div>',
      f'<div class="analysis-section"><h4>Region Focus</h4><p>{analysis.get("region_focus","")}</p></div>',
    ]) if analysis else '<p style="color:#9ca3af">AI analysis unavailable.</p>'}
  </div>

  <!-- 4. OBSERVED INDICATORS -->
  <div class="card">
    <div class="section-label">4 · Observed Indicators</div>
    {indicators_block}
  </div>

  <!-- 5. CONFIDENCE & LIMITATIONS -->
  <div class="card">
    <div class="section-label">5 · Confidence &amp; Limitations</div>
    {'<div class="analysis-section"><h4>Confidence Assessment</h4><p>' + conf_stmt + '</p></div>' if conf_stmt else ''}
    {'<div class="analysis-section"><h4>Limitations</h4><p>' + limitations + '</p></div>' if limitations else ''}
  </div>

  <!-- 6. RECOMMENDATION -->
  <div class="card">
    <div class="section-label">6 · Recommendation</div>
    {'<p>' + recommendation + '</p>' if recommendation else '<p style="color:#9ca3af">No recommendation available.</p>'}
  </div>

  <!-- 7. TECHNICAL METADATA -->
  <details class="card" style="cursor:pointer;">
    <summary style="list-style:none;outline:none;">
      <div class="section-label" style="margin-bottom:0;">
        7 · Technical Metadata &nbsp;
        <span style="font-size:11px;font-weight:400;color:#9ca3af;">(click to expand)</span>
      </div>
    </summary>
    <div style="margin-top:16px;">
      <table class="meta-table">
        {meta_rows}
      </table>
    </div>
  </details>

  <!-- FOOTER -->
  <div class="footer">
    Generated by <strong>ForensIQ</strong> &middot;
    Not a substitute for expert human review &middot;
    Confidence is not a calibrated probability.
  </div>

</div>
</body>
</html>"""

    report_html_path = output_dir / "report.html"
    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[ReportRenderer] report.html saved: {report_html_path}")
    return report_html_path
