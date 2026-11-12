"""Turn a Markdown report into a self-contained, print-ready HTML file, and optionally a PDF.

HTML needs no internet (no external fonts or scripts) and prints cleanly from any browser
("Save as PDF"). Direct PDF output uses Playwright's headless Chromium if it is installed:

    pip install playwright && python -m playwright install chromium
"""
from __future__ import annotations

import html as _html
from pathlib import Path

import markdown

CSS = """
:root{--ink:#14213D;--muted:#5B6575;--rule:#D5DAE1;--wash:#F4F6F8;--pos:#1F7A4D;--neg:#B03A2E;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#fff;color:var(--ink);font:14.5px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
font-variant-numeric:tabular-nums}
main{max-width:980px;margin:0 auto;padding:2.2rem 1.4rem 3rem}
h1,h2,h3,h4{font-family:Georgia,"Times New Roman",serif;font-weight:600;line-height:1.25;break-after:avoid}
h1{font-size:1.9rem;margin:.4rem 0 .6rem;border-bottom:2px solid var(--ink);padding-bottom:.4rem}
h2{font-size:1.35rem;margin:2rem 0 .5rem}
h3{font-size:1.1rem;margin:1.4rem 0 .4rem}
h4{font-size:1rem;margin:1rem 0 .3rem}
p,li{max-width:48em}
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0 1rem;break-inside:auto}
tr{break-inside:avoid}
th{text-align:left;color:var(--muted);font-weight:600;border-bottom:1px solid var(--ink);padding:.4rem .55rem .3rem 0;
white-space:nowrap}
td{padding:.32rem .55rem .32rem 0;border-bottom:1px solid var(--rule)}
tbody tr:nth-child(even){background:var(--wash)}
blockquote{margin:0 0 1rem;padding:.6rem .9rem;border-left:4px solid #C99A1E;background:#FFF6DC}
blockquote p{margin:0}
code{font-size:.85em;background:var(--wash);padding:0 .25rem;border-radius:2px}
a{color:#0E6E6E}
.table-wrap{overflow-x:auto}
footer{margin-top:2.5rem;padding-top:.8rem;border-top:1px solid var(--rule);color:var(--muted);font-size:.8rem}
@media print{
  @page{size:A4;margin:16mm 14mm}
  body{font-size:10.5pt}
  main{max-width:none;padding:0}
  a{color:inherit;text-decoration:none}
  .table-wrap{overflow:visible}
}
"""


def markdown_to_html(md_text: str, title: str, footer: str = "") -> str:
    body = markdown.markdown(md_text, extensions=["tables", "sane_lists"], output_format="html")
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    foot = f"<footer>{_html.escape(footer)}</footer>" if footer else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{_html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}{foot}</main></body></html>")


def write_html(md_path: Path, title: str, footer: str = "") -> Path:
    out = md_path.with_suffix(".html")
    out.write_text(markdown_to_html(md_path.read_text(encoding="utf-8"), title, footer), encoding="utf-8")
    return out


def html_to_pdf(html_path: Path, pdf_path: Path | None = None) -> Path:
    """Print an HTML file to PDF with headless Chromium (Playwright)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("PDF output needs Playwright: pip install playwright && "
                           "python -m playwright install chromium (or open the HTML file and print to PDF)") from exc
    pdf_path = pdf_path or html_path.with_suffix(".pdf")
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Chromium for Playwright is not installed: python -m playwright install chromium "
                               "(or open the HTML file and print to PDF)") from exc
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        page.pdf(path=str(pdf_path), format="A4", print_background=True,
                 margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        browser.close()
    return pdf_path
