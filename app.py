import os
import json
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

# ─── CATEGORY PROMPTS ────────────────────────────────────────────────────────
PROMPTS = {
    "crispr": """You are a cutting-edge biotech research AI for the NEXA BIOTECHNOLOGY group.
Find the 3 NEWEST, most groundbreaking discoveries in CRISPR-Cas9, Cas12, Cas13, base editing,
prime editing, or any gene-editing technology as of 2025-2026.
Focus on: cognitive enhancement, longevity, neuroregeneration, IQ/intelligence gene variants.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"crispr"}]""",

    "nootropics": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find the 3 NEWEST evidence-backed nootropic compounds, stacks, or mechanisms from 2025-2026.
Include: novel compounds, new mechanisms, synergies, advanced stacks.
Focus: neuroplasticity, BDNF, NGF, acetylcholine, dopamine, mTOR, mitochondrial optimization.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"nootropics"}]""",

    "peptides": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find 3 newest, most potent peptide discoveries for cognitive enhancement or neuroregeneration, 2025-2026.
Include: bioregulators (Khavinson), Dihexa, Semax, Selank, BPC-157 variants, TB-500, Epithalon, novel peptides.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"peptides"}]""",

    "longevity": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find 3 newest longevity and anti-aging breakthroughs from 2025-2026.
Include: senolytics, NAD+ pathway, sirtuins, mTOR inhibition, spermidine, urolithin A, clinical trial results.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"longevity"}]""",

    "epigenetic": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find 3 newest epigenetic interventions from 2025-2026 for brain optimization and longevity.
Include: Yamanaka factors (partial reprogramming), epigenetic clocks, methylation modulation,
histone modification, chemical reprogramming.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"epigenetic"}]""",

    "mrna": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find 3 newest mRNA therapeutics, sgRNA applications, or RNA-based technologies from 2025-2026.
Include: self-amplifying mRNA, circular RNA, guide RNA optimization, RNA-based cognitive enhancement.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"mrna"}]""",

    "exosomes": """You are a cutting-edge research AI for the NEXA BIOTECHNOLOGY group.
Find 3 newest exosome/extracellular vesicle discoveries from 2025-2026.
Include: exosome drug delivery, neural exosomes, MSC-derived exosomes for brain repair,
blood-brain barrier crossing, cognitive enhancement applications.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":9,"category":"exosomes"}]""",

    "outofbox": """You are a radical research AI for the NEXA BIOTECHNOLOGY group.
Find 3 completely unconventional discoveries from 2025-2026 for cognitive enhancement or longevity.
Think: ultrasound neuromodulation, photobiomodulation, gut-brain axis hacks, synthetic biology,
xenobiotics, electromagnetic cognitive enhancement, quantum biology, extremophile proteins.

Return ONLY a valid JSON array of 3 objects, nothing else:
[{"title":"...","body":"2-3 sentences of dense scientific insight","tags":["tag1","tag2","tag3"],"noveltyScore":10,"category":"outofbox"}]"""
}


def get_client(api_key: str):
    """Create Gemini client — env var takes priority, then request key."""
    key = os.environ.get("GEMINI_API_KEY") or api_key
    if not key:
        raise ValueError("GEMINI_API_KEY bulunamadı.")
    return genai.Client(api_key=key)


def parse_json_response(text: str) -> list:
    """Extract JSON array from model response robustly."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    return json.loads(match.group(0))


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/research", methods=["POST"])
def research():
    data = request.json or {}
    category = data.get("category", "").lower().strip()
    api_key  = data.get("apiKey", "")

    if category not in PROMPTS:
        return jsonify({"error": f"Geçersiz kategori: {category}"}), 400

    try:
        client = get_client(api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=PROMPTS[category],
        )
        raw = response.text
        findings = parse_json_response(raw)

        # Normalise & validate each finding
        result = []
        for f in findings:
            if not isinstance(f, dict):
                continue
            result.append({
                "title":        str(f.get("title", "")).strip(),
                "body":         str(f.get("body", "")).strip(),
                "tags":         list(f.get("tags", []))[:5],
                "noveltyScore": max(1, min(10, int(f.get("noveltyScore", 7)))),
                "category":     category,
            })

        return jsonify({"findings": result, "raw": raw[:200]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/test", methods=["POST"])
def test_key():
    data    = request.json or {}
    api_key = data.get("apiKey", "")
    try:
        client   = get_client(api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents='Reply with exactly: NEXA_OK',
        )
        ok = "NEXA" in response.text or len(response.text) > 0
        return jsonify({"ok": ok, "response": response.text.strip()[:80]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
