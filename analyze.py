"""
analyze.py — maps glossary terms to their usage in a Markdown report.

For every term in glossary.md it counts, in GenAI_research.md:
  * EXPLICIT mentions — the term itself (abbreviation, plural, grammatical forms)
  * IMPLICIT mentions — full names, synonyms, related phrases (from aliases.json)

It then builds a graph:
  * report node  -> term node   (edge width = total mentions)
  * term  <-> term  "co-occurrence": both terms appear in the same paragraph
  * term  ->  term  "definition": the glossary definition of one term mentions another

Outputs (in the output folder):
  counts.csv   — explicit / implicit / total per term + sections where it appears
  matches.csv  — every single match (for manual checking)
  graph.html   — interactive graph (vis-network, opens in any browser)
  graph.png    — static picture: network + explicit/implicit bar chart

Usage:
  python analyze.py
  python analyze.py --glossary glossary.md --report GenAI_research.md --aliases aliases.json --out output
"""

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to file, no window needed
import matplotlib.pyplot as plt
import networkx as nx

# Colors (validated categorical slots 1–2 + neutrals)
EXPLICIT_COLOR = "#2a78d6"  # blue
IMPLICIT_COLOR = "#eb6834"  # orange
REPORT_COLOR = "#52514e"    # neutral gray for the report hub
COOC_EDGE = "#b5b4ad"
DEF_EDGE = "#eb6834"


# ----------------------------------------------------------------------------
# 1. Parsing
# ----------------------------------------------------------------------------
def parse_glossary(path: Path) -> dict:
    """Return {term: definition}. Handles lines like '• **Term**: definition'
    and the malformed '• **AI-Native Tool**s: ...' (letters glued after the bold)."""
    terms = {}
    pattern = re.compile(r"\*\*(.+?)\*\*([A-Za-z]*)\s*:\s*(.+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            term = (m.group(1) + m.group(2)).strip()
            terms[term] = m.group(3).strip()
    return terms


def clean_markdown(text: str) -> str:
    """Remove Markdown noise that would create false matches or break word boundaries."""
    text = html.unescape(text)                          # &#x20; -> space
    text = re.sub(r"\]\([^)]*\)", "]", text)            # drop link URLs: [name](url) -> [name]
    text = re.sub(r"https?://\S+", " ", text)           # drop bare URLs
    text = re.sub(r"\\([^\w\s])", r"\1", text)          # \--- -> ---, 1\. -> 1.
    text = text.replace("**", "").replace("__", "")     # bold markers
    return text


def split_paragraphs(text: str) -> list:
    """Return a list of {'section': str, 'text': str}. A heading starts a new section."""
    paragraphs = []
    section = "(intro)"
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or set(block) <= set("-|: "):
            continue
        heading = re.match(r"^(#{1,6})\s*(.+)", block)
        if heading and len(heading.group(1)) <= 2:
            section = heading.group(2).strip()
        paragraphs.append({"section": section, "text": block})
    return paragraphs


def load_aliases(path: Path, terms: dict) -> dict:
    """Return {term: {'explicit': [...], 'implicit': [...]}} for every glossary term."""
    raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    by_lower = {k.lower(): v for k, v in raw.items() if not k.startswith("_")}
    aliases = {}
    for term in terms:
        entry = by_lower.get(term.lower(), {})
        explicit = entry.get("explicit") or [term]
        implicit = entry.get("implicit", [])
        aliases[term] = {"explicit": explicit, "implicit": implicit}
        if term.lower() not in by_lower:
            print(f"  ! '{term}' not in aliases.json — using the term itself as the only explicit form")
    return aliases


# ----------------------------------------------------------------------------
# 2. Matching
# ----------------------------------------------------------------------------
def compile_alias(alias: str) -> re.Pattern:
    """Whole-phrase match. ALL-CAPS abbreviations are case-sensitive, the rest are not."""
    is_abbrev = alias.upper() == alias and any(c.isalpha() for c in alias)
    flags = 0 if is_abbrev else re.IGNORECASE
    return re.compile(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", flags)


def find_mentions(text: str, forms: dict) -> list:
    """Return [(kind, alias, matched_text, start, end)].
    Explicit forms first, then implicit; longer phrases first; a span is never counted twice
    for the same term (so 'context window' is not also counted as 'window')."""
    claimed = []
    found = []
    for kind in ("explicit", "implicit"):
        for alias in sorted(forms[kind], key=len, reverse=True):
            for m in compile_alias(alias).finditer(text):
                s, e = m.span()
                if any(s < ce and e > cs for cs, ce in claimed):
                    continue
                claimed.append((s, e))
                found.append((kind, alias, m.group(0), s, e))
    return found


# ----------------------------------------------------------------------------
# 3. Analysis
# ----------------------------------------------------------------------------
def analyze(terms: dict, aliases: dict, paragraphs: list) -> dict:
    stats = {t: {"explicit": 0, "implicit": 0, "sections": Counter(), "aliases": Counter()} for t in terms}
    matches = []
    cooccurrence = Counter()

    for i, para in enumerate(paragraphs, start=1):
        present = set()
        for term in terms:
            for kind, alias, matched, _, _ in find_mentions(para["text"], aliases[term]):
                stats[term][kind] += 1
                stats[term]["sections"][para["section"]] += 1
                stats[term]["aliases"][f"{matched.lower()} ({kind[:3]})"] += 1
                matches.append({"term": term, "type": kind, "matched_text": matched,
                                "alias": alias, "paragraph": i, "section": para["section"]})
                present.add(term)
        for a, b in combinations(sorted(present), 2):
            cooccurrence[(a, b)] += 1

    # Glossary definitions that mention other terms
    definition_links = []
    for term, definition in terms.items():
        for other in terms:
            if other != term and find_mentions(definition, aliases[other]):
                definition_links.append((term, other))

    return {"stats": stats, "matches": matches, "cooccurrence": cooccurrence,
            "definition_links": definition_links}


def build_graph(terms: dict, result: dict, report_name: str) -> nx.Graph:
    g = nx.Graph()
    g.add_node(report_name, kind="report")
    for term in terms:
        s = result["stats"][term]
        total = s["explicit"] + s["implicit"]
        g.add_node(term, kind="term", explicit=s["explicit"], implicit=s["implicit"], total=total)
        if total:
            g.add_edge(report_name, term, kind="usage", weight=total)
    for (a, b), w in result["cooccurrence"].items():
        g.add_edge(a, b, kind="cooccurrence", weight=w)
    for a, b in result["definition_links"]:
        if g.has_edge(a, b):
            g.edges[a, b]["definition"] = f"{a} → {b}"
        else:
            g.add_edge(a, b, kind="definition", weight=1, definition=f"{a} → {b}")
    return g


# ----------------------------------------------------------------------------
# 4. Outputs
# ----------------------------------------------------------------------------
def write_csvs(terms: dict, result: dict, out: Path) -> list:
    rows = []
    for term in terms:
        s = result["stats"][term]
        rows.append({
            "term": term,
            "explicit": s["explicit"],
            "implicit": s["implicit"],
            "total": s["explicit"] + s["implicit"],
            "sections": "; ".join(f"{sec} ({n})" for sec, n in s["sections"].most_common()),
            "matched_forms": "; ".join(f"{a} ×{n}" for a, n in s["aliases"].most_common()),
        })
    rows.sort(key=lambda r: r["total"], reverse=True)
    with open(out / "counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(out / "matches.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["term", "type", "matched_text", "alias", "paragraph", "section"])
        w.writeheader()
        w.writerows(result["matches"])
    return rows


def draw_png(g: nx.Graph, rows: list, report_name: str, path: Path) -> None:
    fig, (ax_net, ax_bar) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.patch.set_facecolor("#fcfcfb")

    # --- network ---
    pos = nx.spring_layout(g, weight="weight", seed=42, k=1.1)
    pos[report_name] = (0, 0)
    pos = nx.spring_layout(g, pos=pos, fixed=[report_name], weight="weight", seed=42, k=1.1)

    usage = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "usage"]
    cooc = [(u, v) for u, v, d in g.edges(data=True) if d["kind"] == "cooccurrence"]
    defs = [(u, v) for u, v, d in g.edges(data=True) if "definition" in d]

    nx.draw_networkx_edges(g, pos, edgelist=cooc, ax=ax_net, edge_color=COOC_EDGE, alpha=0.6,
                           width=[0.5 + g.edges[e]["weight"] * 0.35 for e in cooc])
    nx.draw_networkx_edges(g, pos, edgelist=usage, ax=ax_net, edge_color=EXPLICIT_COLOR, alpha=0.25,
                           width=[0.5 + g.edges[e]["weight"] * 0.12 for e in usage])
    nx.draw_networkx_edges(g, pos, edgelist=defs, ax=ax_net, edge_color=DEF_EDGE, width=2, style="dashed")

    terms = [n for n, d in g.nodes(data=True) if d["kind"] == "term"]
    nx.draw_networkx_nodes(g, pos, nodelist=terms, ax=ax_net, node_color=EXPLICIT_COLOR,
                           edgecolors="#fcfcfb", linewidths=2,
                           node_size=[300 + g.nodes[n]["total"] * 40 for n in terms])
    nx.draw_networkx_nodes(g, pos, nodelist=[report_name], ax=ax_net, node_color=REPORT_COLOR,
                           node_shape="s", node_size=1800)
    labels = {n: f"{n}\n{g.nodes[n]['explicit']} exp / {g.nodes[n]['implicit']} imp" for n in terms}
    labels[report_name] = "report"
    for n, (x, y) in pos.items():
        ax_net.text(x, y - 0.09, labels[n], ha="center", va="top", fontsize=9, color="#0b0b0b",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fcfcfb", ec="none", alpha=0.85))
    ax_net.set_title("Glossary terms in the report\nnode size = total mentions · gray = same paragraph · "
                     "dashed orange = glossary definition link", fontsize=11, color="#52514e")
    ax_net.axis("off")

    # --- stacked bar ---
    rows_sorted = sorted(rows, key=lambda r: r["total"])
    names = [r["term"] for r in rows_sorted]
    exp = [r["explicit"] for r in rows_sorted]
    imp = [r["implicit"] for r in rows_sorted]
    ax_bar.barh(names, exp, color=EXPLICIT_COLOR, label="Explicit", edgecolor="#fcfcfb", linewidth=2)
    ax_bar.barh(names, imp, left=exp, color=IMPLICIT_COLOR, label="Implicit", edgecolor="#fcfcfb", linewidth=2)
    for i, r in enumerate(rows_sorted):
        ax_bar.text(r["total"] + 0.4, i, f"{r['explicit']} + {r['implicit']} = {r['total']}",
                    va="center", fontsize=9, color="#52514e")
    ax_bar.set_title("Explicit vs implicit mentions", fontsize=11, color="#52514e")
    ax_bar.legend(loc="lower right", frameon=False)
    ax_bar.set_xlim(0, max(r["total"] for r in rows) * 1.35 + 1)
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)
    ax_bar.set_facecolor("#fcfcfb")
    ax_bar.tick_params(colors="#52514e")
    ax_bar.grid(axis="x", color="#e5e4df", linewidth=0.8)
    ax_bar.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Glossary terms in the GenAI report</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root { color-scheme: light; --surface:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --grid:#e5e4df;
          --explicit:#2a78d6; --implicit:#eb6834; --report:#52514e; --cooc:#b5b4ad; }
  @media (prefers-color-scheme: dark) {
    :root { color-scheme: dark; --surface:#1a1a19; --text:#f0efec; --muted:#c3c2b7; --grid:#33322f;
            --explicit:#3987e5; --implicit:#d95926; --report:#c3c2b7; --cooc:#5c5b56; }
  }
  body { margin:0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--surface); color:var(--text); }
  main { max-width:1200px; margin:0 auto; padding:24px; }
  h1 { font-size:22px; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 16px; }
  #graph { height:640px; border:1px solid var(--grid); border-radius:8px; }
  .legend { display:flex; gap:20px; flex-wrap:wrap; color:var(--muted); font-size:13px; margin:10px 0 24px; }
  .legend span::before { content:""; display:inline-block; width:18px; height:0; margin-right:6px; vertical-align:middle; }
  .l-usage::before { border-top:3px solid var(--explicit); opacity:.5; }
  .l-cooc::before  { border-top:3px solid var(--cooc); }
  .l-def::before   { border-top:3px dashed var(--implicit); }
  table { border-collapse:collapse; width:100%; font-size:14px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--grid); vertical-align:top; }
  th { color:var(--muted); font-weight:600; }
  td.num { text-align:right; font-variant-numeric: tabular-nums; }
  .bar { display:flex; height:10px; min-width:4px; }
  .bar i { display:block; height:100%; }
  .bar i + i { margin-left:2px; }
  .bar i:first-child { border-radius:4px 0 0 4px; }
  .bar i:last-child { border-radius:0 4px 4px 0; }
  .forms { color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<main>
  <h1>Glossary terms in <code>__REPORT__</code></h1>
  <p class="sub">Node size = total mentions. Hover a node for explicit / implicit counts. Drag nodes to rearrange.</p>
  <div id="graph"></div>
  <div class="legend">
    <span class="l-usage">report → term (width = mentions)</span>
    <span class="l-cooc">terms in the same paragraph (width = paragraphs)</span>
    <span class="l-def">glossary definition mentions the other term</span>
  </div>
  <h2 style="font-size:18px">Counts</h2>
  <table>
    <thead><tr><th>Term</th><th class="num">Explicit</th><th class="num">Implicit</th><th class="num">Total</th>
      <th style="width:22%"><span style="color:var(--explicit)">■</span> explicit <span style="color:var(--implicit)">■</span> implicit</th>
      <th>Matched forms</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const C = { explicit: css('--explicit'), implicit: css('--implicit'), report: css('--report'),
            cooc: css('--cooc'), text: css('--text'), surface: css('--surface') };

const nodes = DATA.nodes.map(n => n.kind === 'report'
  ? { id: n.id, label: 'REPORT', shape: 'box', color: { background: C.report, border: C.report },
      font: { color: C.surface, size: 14 }, title: n.id, mass: 3 }
  : { id: n.id, label: `${n.id}\\n${n.explicit} / ${n.implicit}`, shape: 'dot',
      size: 10 + Math.sqrt(n.total) * 6,
      color: { background: C.explicit, border: C.surface, highlight: { background: C.implicit, border: C.surface } },
      borderWidth: 2, font: { color: C.text, size: 13, multi: false },
      title: `${n.id}\\nExplicit: ${n.explicit}\\nImplicit: ${n.implicit}\\nTotal: ${n.total}` });

const edges = DATA.edges.map(e => {
  if (e.kind === 'usage') return { from: e.source, to: e.target, width: 1 + e.weight * 0.25,
      color: { color: C.explicit, opacity: 0.3 }, title: `${e.target}: ${e.weight} mentions in the report` };
  if (e.kind === 'definition') return { from: e.source, to: e.target, width: 2, dashes: true, arrows: 'to',
      color: { color: C.implicit }, title: `Glossary: ${e.definition}` };
  return { from: e.source, to: e.target, width: 0.5 + e.weight * 0.6, dashes: e.definition ? [6, 4] : false,
      color: { color: e.definition ? C.implicit : C.cooc, opacity: 0.8 },
      title: `${e.source} + ${e.target}: together in ${e.weight} paragraph(s)` + (e.definition ? `\\nGlossary: ${e.definition}` : '') };
});

new vis.Network(document.getElementById('graph'),
  { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) },
  { physics: { solver: 'forceAtlas2Based', forceAtlas2Based: { gravitationalConstant: -80, springLength: 140 },
               stabilization: { iterations: 300 } },
    interaction: { hover: true, tooltipDelay: 80 } });

const max = Math.max(...DATA.rows.map(r => r.total), 1);
document.getElementById('rows').innerHTML = DATA.rows.map(r => `
  <tr><td>${r.term}</td><td class="num">${r.explicit}</td><td class="num">${r.implicit}</td><td class="num"><b>${r.total}</b></td>
  <td><div class="bar" style="width:${r.total / max * 100}%">
    ${r.explicit ? `<i style="flex:${r.explicit};background:${C.explicit}"></i>` : ''}
    ${r.implicit ? `<i style="flex:${r.implicit};background:${C.implicit}"></i>` : ''}</div></td>
  <td class="forms">${r.matched_forms}</td></tr>`).join('');
</script>
</body>
</html>
"""


def write_html(g: nx.Graph, rows: list, report_name: str, path: Path) -> None:
    data = {
        "nodes": [{"id": n, **d} for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in g.edges(data=True)],
        "rows": rows,
    }
    page = (HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
                         .replace("__REPORT__", html.escape(report_name)))
    path.write_text(page, encoding="utf-8")


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Glossary term usage graph")
    parser.add_argument("--glossary", default="glossary.md")
    parser.add_argument("--report", default="GenAI_research.md")
    parser.add_argument("--aliases", default="aliases.json")
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    terms = parse_glossary(Path(args.glossary))
    print(f"Glossary: {len(terms)} terms")
    aliases = load_aliases(Path(args.aliases), terms)

    report_name = Path(args.report).name
    paragraphs = split_paragraphs(clean_markdown(Path(args.report).read_text(encoding="utf-8")))
    print(f"Report: {len(paragraphs)} paragraphs")

    result = analyze(terms, aliases, paragraphs)
    graph = build_graph(terms, result, report_name)
    rows = write_csvs(terms, result, out)
    draw_png(graph, rows, report_name, out / "graph.png")
    write_html(graph, rows, report_name, out / "graph.html")

    print(f"\n{'Term':<20}{'Explicit':>9}{'Implicit':>9}{'Total':>7}")
    for r in rows:
        print(f"{r['term']:<20}{r['explicit']:>9}{r['implicit']:>9}{r['total']:>7}")
    print(f"\nSaved to {out.resolve()}: counts.csv, matches.csv, graph.png, graph.html")


if __name__ == "__main__":
    main()
