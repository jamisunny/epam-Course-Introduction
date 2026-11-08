python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install networkx pyvis matplotlib

# macOS / Linux
source .venv/bin/activate
pip install networkx pyvis matplotlib
-----

python analyze.py
-----

the results in output folder
-----

the algorithm scheme:
 
glossary.md ──► 1. Glossary parsing ──► list of terms
GenAI_research.md ──► 1. Cleaning and text chunking (paragraph splitting)
aliases.json ──► 1. Synonyms for each term
        ↓
2. Mention extraction (explicit and implicit) per paragraph
        ↓
3. Statistics and co-occurrence (term relationship) calculations
        ↓
4. Graph construction (networkx)
        ↓
5. Exporting results: counts.csv, matches.csv, graph.png, graph.html

-----

commands to git copilot. Switch it to agent mode

Create a Python script analyze.py that:
1. Parses glossary.md. Each term is in the format "• **Term**: definition"
   (note: one line is malformed: "**AI-Native Tool**s" — handle it).
2. Reads GenAI_research.md and splits it into paragraphs (and records which ## section each belongs to).
3. For each term counts:
   - EXPLICIT mentions: the exact term or abbreviation, case-insensitive,
     whole words, including plural forms (token/tokens, LLM/LLMs, hallucination/hallucinations).
   - IMPLICIT mentions: full names, synonyms and related phrases from an
     ALIASES dictionary in a separate file aliases.json
     (e.g. RAG -> "retrieval-augmented generation", "retrieval layer", "vector database";
      ML -> "machine learning"; Hallucinations -> "fabricate", "false output").
     Do not count a match twice if it overlaps with an explicit mention.
4. Builds a graph with networkx:
   - one central node for the report, one node per term;
   - node size = total mentions; tooltip shows explicit / implicit counts;
   - edges term–term when both appear in the same paragraph, weight = number of such paragraphs;
   - edges term–term when one term's glossary definition mentions another term.
5. Saves graph.html (pyvis, interactive), graph.png (matplotlib) and counts.csv.
Then run the script and fix any errors.