"""
NEXA BIOTECH — ULTRA AGENT SYSTEM
4-agent pipeline: SCOUT → CRITIC → SYNTHESIZER → EVOLVER
Self-evolving prompts, knowledge graph, distillation engine
"""
import os, json, re, time, logging, threading, uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from google import genai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("NEXA")

app  = Flask(__name__)
CORS(app)

# ─── MODELS (fallback chain) ──────────────────────────────────────────────────
MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
]

# ─── SHARED STATE (in-memory + persisted to JSON) ─────────────────────────────
DATA_FILE = "nexa_data.json"

def default_state():
    return {
        "findings":       [],   # all research findings
        "insights":       [],   # synthesized compound insights
        "connections":    [],   # edges between findings
        "prompt_genome":  {},   # evolved prompts per category
        "evolution_log":  [],   # history of prompt evolutions
        "agent_log":      [],   # global event log (SSE streamed)
        "cycle_count":    0,
        "distill_count":  0,
    }

def load_state():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                s = json.load(f)
                # merge missing keys
                d = default_state()
                for k, v in d.items():
                    if k not in s:
                        s[k] = v
                return s
        except: pass
    return default_state()

def save_state():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(STATE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error("save_state: %s", e)

STATE = load_state()
STATE_LOCK = threading.Lock()

# ─── SSE EVENT BUS ────────────────────────────────────────────────────────────
_subscribers: list[list] = []
_sub_lock = threading.Lock()

def sse_emit(event_type: str, data: dict):
    """Broadcast an SSE event to all connected clients."""
    payload = json.dumps({"type": event_type, "data": data, "ts": _ts()})
    with _sub_lock:
        dead = []
        for q in _subscribers:
            try: q.append(payload)
            except: dead.append(q)
        for q in dead:
            _subscribers.remove(q)

def _ts():
    return datetime.now().strftime("%H:%M:%S")

# ─── GEMINI WRAPPER ───────────────────────────────────────────────────────────
MAX_RETRY = 3
BASE_WAIT  = 18

def _is_quota(e): return "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)

def _retry_delay(e):
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)", str(e))
    return min(float(m.group(1)) + 3, 120) if m else BASE_WAIT

def gemini_call(client, prompt: str, model_hint: str = None) -> tuple[str, str]:
    chain = [model_hint] + [m for m in MODELS if m != model_hint] if model_hint else MODELS
    last_err = None
    for model in chain:
        backoff = BASE_WAIT
        for attempt in range(1, MAX_RETRY + 1):
            try:
                resp = client.models.generate_content(model=model, contents=prompt)
                return resp.text, model
            except Exception as e:
                last_err = e
                if _is_quota(e):
                    wait = _retry_delay(e) if attempt == 1 else backoff
                    backoff = min(backoff * 2, 120)
                    sse_emit("retry", {"model": model, "wait": int(wait), "attempt": attempt})
                    time.sleep(wait)
                else:
                    break
    raise RuntimeError(f"All models failed. Last: {str(last_err)[:200]}")

def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not m: return None
    return json.loads(m.group(1))

# ─── INITIAL PROMPT GENOME ────────────────────────────────────────────────────
BASE_PROMPTS = {
    "crispr": "Find 3 newest CRISPR-Cas9/12/13, base editing, prime editing breakthroughs 2025-2026 for cognitive enhancement, IQ genes, neuroregeneration, longevity.",
    "nootropics": "Find 3 newest nootropic compounds, stacks, or mechanisms 2025-2026. Focus: neuroplasticity, BDNF, NGF, mTOR, mitochondrial optimization, synaptic enhancement.",
    "peptides": "Find 3 newest research peptides 2025-2026: bioregulators (Khavinson), Dihexa, Semax, Selank, BPC-157 variants, Epithalon, novel neuropeptides.",
    "longevity": "Find 3 newest longevity breakthroughs 2025-2026: senolytics, NAD+ pathway, sirtuins, mTOR inhibitors, spermidine, urolithin A, clinical trial results.",
    "epigenetic": "Find 3 newest epigenetic interventions 2025-2026: partial Yamanaka reprogramming, epigenetic clocks, DNA methylation modulation, chemical reprogramming.",
    "mrna": "Find 3 newest mRNA/sgRNA/RNA-based technologies 2025-2026: self-amplifying mRNA, circular RNA, RNA-based cognitive enhancement, novel delivery.",
    "exosomes": "Find 3 newest exosome discoveries 2025-2026: neural exosomes, MSC-derived brain repair, BBB crossing, cargo engineering, cognitive enhancement.",
    "outofbox": "Find 3 radical out-of-the-box discoveries 2025-2026 for cognition/longevity: ultrasound neuromodulation, photobiomodulation, gut-brain hacks, xenobiotics, quantum biology.",
}

def get_genome(category: str) -> str:
    return STATE["prompt_genome"].get(category, BASE_PROMPTS.get(category, ""))

# ─── AGENT DEFINITIONS ────────────────────────────────────────────────────────

SCOUT_TEMPLATE = """You are SCOUT, an elite biotech research AI for NEXA BIOTECHNOLOGY.

MISSION: {mission}

PREVIOUSLY KNOWN (do NOT repeat these concepts):
{known}

EVOLVED RESEARCH DIRECTION (follow this):
{genome}

Your task: Find exactly 3 NEW, high-novelty findings not covered by known items above.
Be specific, scientific, dense. Include mechanism of action, compound names, study results.

Return ONLY valid JSON array:
[{{
  "title": "specific descriptive title",
  "body": "2-3 sentences: mechanism, evidence, significance",
  "tags": ["tag1","tag2","tag3","tag4"],
  "noveltyScore": 9,
  "significance": "brief note on why this matters for IQ/longevity",
  "category": "{category}",
  "connections": ["concept1","concept2"]
}}]"""

CRITIC_TEMPLATE = """You are CRITIC, a rigorous scientific validator.

Evaluate these research findings for NEXA BIOTECHNOLOGY:
{findings}

For each finding:
1. Verify it's genuinely novel (not just rehashing old info)
2. Check scientific plausibility
3. Rate quality 1-10

Return ONLY valid JSON array — keep only findings scoring 6+, improve their body text:
[{{
  "title": "...",
  "body": "improved, more precise description",
  "tags": [...],
  "noveltyScore": <1-10>,
  "qualityScore": <1-10>,
  "significance": "...",
  "category": "...",
  "connections": [...]
}}]"""

SYNTHESIZER_TEMPLATE = """You are SYNTHESIZER, a knowledge integration engine for NEXA BIOTECHNOLOGY.

You have these recent findings across categories:
{findings}

TASK: Generate 2 compound insights by finding NON-OBVIOUS connections between findings.
Each insight should connect at least 2 different categories and reveal a deeper pattern.

Return ONLY valid JSON array:
[{{
  "title": "Cross-domain insight title",
  "body": "3-4 sentences explaining the connection and its implications for cognition/longevity",
  "connected_findings": ["finding_title_1", "finding_title_2"],
  "categories": ["cat1","cat2"],
  "tags": ["tag1","tag2","tag3"],
  "noveltyScore": 9,
  "type": "synthesis"
}}]"""

EVOLVER_TEMPLATE = """You are EVOLVER, a meta-learning AI that improves research strategies for NEXA BIOTECHNOLOGY.

CYCLE {cycle} ANALYSIS:
Current genome for {category}: "{current_genome}"
Recent findings in this category: {recent}
Global top tags across all research: {top_tags}
Gaps detected (topics not yet found): {gaps}

TASK: Write an IMPROVED research directive for {category} that:
1. Builds on what was discovered (go deeper on promising threads)
2. Avoids already-known territory
3. Pushes toward unknown frontiers
4. Is more specific and targeted than before

Return ONLY a JSON object:
{{"evolved_prompt": "new directive text (2-3 sentences, very specific)", "reasoning": "why this evolution", "focus_shift": "what changed"}}"""

DISTILLER_TEMPLATE = """You are DISTILLER, a knowledge crystallization engine for NEXA BIOTECHNOLOGY.

You have accumulated {count} findings. Distill them into the top 5 MASTER INSIGHTS — 
the highest-order patterns, principles, and actionable protocols emerging from all research.

ALL FINDINGS:
{all_findings}

Return ONLY valid JSON array of 5 master insights:
[{{
  "title": "Master Insight title",
  "body": "4-5 sentences: the core pattern, evidence base, and practical implications",
  "protocol_hint": "actionable next step",
  "confidence": <1-10>,
  "categories_spanned": ["cat1","cat2"],
  "tags": ["tag1","tag2","tag3"]
}}]"""

# ─── AGENT PIPELINE ───────────────────────────────────────────────────────────

class AgentPipeline:

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or api_key)
        if not (os.environ.get("GEMINI_API_KEY") or api_key):
            raise ValueError("GEMINI_API_KEY bulunamadı")

    def _emit(self, agent: str, msg: str, level: str = "info"):
        sse_emit("agent_log", {"agent": agent, "msg": msg, "level": level})
        log.info("[%s] %s", agent, msg)

    # ── SCOUT ──────────────────────────────────────────────────────────────────
    def scout(self, category: str) -> list:
        self._emit("SCOUT", f"→ {category.upper()} taranıyor...", "scan")

        # build known context (last 10 titles in this category)
        with STATE_LOCK:
            known = [f["title"] for f in STATE["findings"] if f.get("category") == category][-10:]

        genome = get_genome(category)
        prompt = SCOUT_TEMPLATE.format(
            mission=f"Research the {category} domain for NEXA BIOTECHNOLOGY",
            known=json.dumps(known) if known else "none yet",
            genome=genome,
            category=category,
        )

        try:
            raw, model = gemini_call(self.client, prompt)
            self._emit("SCOUT", f"✓ {category} — model: {model}", "success")
            parsed = parse_json(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception as e:
            self._emit("SCOUT", f"✗ {category}: {str(e)[:80]}", "error")
            return []

    # ── CRITIC ─────────────────────────────────────────────────────────────────
    def critic(self, findings: list) -> list:
        if not findings:
            return []
        self._emit("CRITIC", f"Evaluating {len(findings)} findings...", "info")
        prompt = CRITIC_TEMPLATE.format(findings=json.dumps(findings, ensure_ascii=False))
        try:
            raw, model = gemini_call(self.client, prompt)
            result = parse_json(raw)
            if isinstance(result, list):
                kept = [f for f in result if isinstance(f, dict) and f.get("qualityScore", 6) >= 6]
                self._emit("CRITIC", f"✓ {len(kept)}/{len(findings)} geçti — model: {model}", "success")
                return kept
        except Exception as e:
            self._emit("CRITIC", f"✗ Critic failed: {str(e)[:80]}", "error")
        return findings  # fallback: keep all

    # ── SYNTHESIZER ────────────────────────────────────────────────────────────
    def synthesize(self, recent_findings: list) -> list:
        if len(recent_findings) < 4:
            return []
        self._emit("SYNTH", f"Synthesizing {len(recent_findings)} findings...", "info")
        prompt = SYNTHESIZER_TEMPLATE.format(
            findings=json.dumps(recent_findings[-12:], ensure_ascii=False)
        )
        try:
            raw, model = gemini_call(self.client, prompt)
            result = parse_json(raw)
            if isinstance(result, list):
                self._emit("SYNTH", f"✓ {len(result)} compound insight — model: {model}", "success")
                return result
        except Exception as e:
            self._emit("SYNTH", f"✗ Synthesis failed: {str(e)[:80]}", "error")
        return []

    # ── EVOLVER ────────────────────────────────────────────────────────────────
    def evolve(self, category: str):
        self._emit("EVOLVER", f"Evolving genome for {category}...", "info")
        with STATE_LOCK:
            recent = [f["title"] for f in STATE["findings"] if f.get("category") == category][-6:]
            all_tags = [t for f in STATE["findings"] for t in f.get("tags", [])]
            top_tags = sorted(set(all_tags), key=lambda t: all_tags.count(t), reverse=True)[:15]
            # detect gaps — topics mentioned in connections but not yet explored
            all_connections = [c for f in STATE["findings"] for c in f.get("connections", [])]
            explored_tags = set(all_tags)
            gaps = [c for c in all_connections if c not in explored_tags][:8]

        current_genome = get_genome(category)
        prompt = EVOLVER_TEMPLATE.format(
            cycle=STATE.get("cycle_count", 0),
            category=category,
            current_genome=current_genome,
            recent=json.dumps(recent),
            top_tags=json.dumps(top_tags),
            gaps=json.dumps(gaps),
        )
        try:
            raw, model = gemini_call(self.client, prompt)
            result = parse_json(raw)
            if isinstance(result, dict) and "evolved_prompt" in result:
                new_genome = result["evolved_prompt"]
                with STATE_LOCK:
                    STATE["prompt_genome"][category] = new_genome
                    STATE["evolution_log"].append({
                        "id": str(uuid.uuid4())[:8],
                        "category": category,
                        "old": current_genome[:100],
                        "new": new_genome[:200],
                        "reasoning": result.get("reasoning", ""),
                        "focus_shift": result.get("focus_shift", ""),
                        "cycle": STATE["cycle_count"],
                        "ts": _ts(),
                    })
                self._emit("EVOLVER", f"✓ {category} genome evolved (gen {STATE['cycle_count']}) — model: {model}", "success")
                sse_emit("genome_evolved", {
                    "category": category,
                    "new_genome": new_genome[:150],
                    "reasoning": result.get("reasoning",""),
                })
        except Exception as e:
            self._emit("EVOLVER", f"✗ Evolution failed: {str(e)[:80]}", "error")

    # ── DISTILLER ──────────────────────────────────────────────────────────────
    def distill(self):
        with STATE_LOCK:
            total = len(STATE["findings"])
        if total < 8:
            return
        self._emit("DISTIL", f"Distilling {total} findings into master insights...", "info")
        sse_emit("distilling", {"count": total})

        with STATE_LOCK:
            summaries = [{"title": f["title"], "category": f.get("category"), "body": f.get("body","")} for f in STATE["findings"]]

        prompt = DISTILLER_TEMPLATE.format(
            count=total,
            all_findings=json.dumps(summaries, ensure_ascii=False),
        )
        try:
            raw, model = gemini_call(self.client, prompt)
            result = parse_json(raw)
            if isinstance(result, list):
                distilled = [{
                    **i,
                    "id": str(uuid.uuid4())[:8],
                    "type": "distilled",
                    "timestamp": datetime.now().isoformat(),
                } for i in result if isinstance(i, dict)]

                with STATE_LOCK:
                    STATE["insights"] = distilled  # replace with latest distillation
                    STATE["distill_count"] += 1

                save_state()
                self._emit("DISTIL", f"✓ {len(distilled)} master insights — model: {model}", "success")
                sse_emit("distilled", {"insights": distilled, "count": len(distilled)})
        except Exception as e:
            self._emit("DISTIL", f"✗ Distillation failed: {str(e)[:80]}", "error")

    # ── FULL CYCLE ─────────────────────────────────────────────────────────────
    def run_cycle(self, categories: list):
        with STATE_LOCK:
            STATE["cycle_count"] += 1
            cycle = STATE["cycle_count"]

        sse_emit("cycle_start", {"cycle": cycle, "categories": categories})
        self._emit("SYSTEM", f"━━ CYCLE {cycle} BAŞLIYOR → {len(categories)} kategori ━━", "success")

        all_new = []
        for i, cat in enumerate(categories):
            sse_emit("progress", {"done": i, "total": len(categories), "current": cat})

            # 1. SCOUT
            raw_findings = self.scout(cat)
            if not raw_findings:
                time.sleep(2)
                continue

            # 2. CRITIC
            time.sleep(1)
            validated = self.critic(raw_findings)

            # 3. Dedup & store
            new_count = 0
            with STATE_LOCK:
                existing_titles = {_norm(f["title"]) for f in STATE["findings"]}
                for f in validated:
                    title = str(f.get("title","")).strip()
                    if not title: continue
                    if _norm(title) in existing_titles: continue
                    if _is_dup(title, existing_titles): continue
                    entry = {
                        **f,
                        "id": str(uuid.uuid4())[:12],
                        "timestamp": datetime.now().isoformat(),
                        "cycle": cycle,
                    }
                    STATE["findings"].insert(0, entry)
                    existing_titles.add(_norm(title))
                    new_count += 1
                    sse_emit("new_finding", entry)

            all_new.extend(validated[:new_count if new_count else 0])
            self._emit("SYSTEM", f"  {cat}: +{new_count} yeni bulgu", "success")

            # 4. EVOLVER (after each category)
            time.sleep(1)
            self.evolve(cat)
            time.sleep(2)

        # 5. SYNTHESIZER (after all categories)
        sse_emit("progress", {"done": len(categories), "total": len(categories), "current": "synthesis"})
        time.sleep(1)
        with STATE_LOCK:
            recent = STATE["findings"][:20]

        new_insights = self.synthesize(recent)
        if new_insights:
            with STATE_LOCK:
                for ins in new_insights:
                    ins["id"] = str(uuid.uuid4())[:12]
                    ins["timestamp"] = datetime.now().isoformat()
                    ins["cycle"] = cycle
                    STATE["insights"].insert(0, ins)
                    sse_emit("new_insight", ins)

        # 6. DISTILLER (every 3 cycles)
        if cycle % 3 == 0:
            time.sleep(1)
            self.distill()

        save_state()

        with STATE_LOCK:
            total = len(STATE["findings"])
            ins_total = len(STATE["insights"])

        sse_emit("cycle_complete", {
            "cycle": cycle,
            "total_findings": total,
            "total_insights": ins_total,
            "new_this_cycle": len(all_new),
        })
        self._emit("SYSTEM", f"━━ CYCLE {cycle} TAMAMLANDI → {total} toplam bulgu, {ins_total} insight ━━", "success")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def _is_dup(title: str, existing: set) -> bool:
    words = [w for w in _norm(title).split() if len(w) > 3]
    if not words: return False
    for ex in existing:
        ex_words = [w for w in ex.split() if len(w) > 3]
        overlap = sum(1 for w in words if w in ex_words)
        if ex_words and overlap / len(words) > 0.55:
            return True
    return False


# ─── BACKGROUND RUNNER ────────────────────────────────────────────────────────
_bg_thread = None
_bg_stop   = threading.Event()
_current_pipeline: AgentPipeline | None = None

def _bg_loop(pipeline: AgentPipeline, categories: list, interval_min: int):
    while not _bg_stop.is_set():
        try:
            pipeline.run_cycle(categories)
        except Exception as e:
            sse_emit("agent_log", {"agent":"SYSTEM","msg": f"Cycle error: {str(e)[:100]}","level":"error"})
        # wait interval_min minutes or until stopped
        for _ in range(interval_min * 60):
            if _bg_stop.is_set(): return
            time.sleep(1)


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    with STATE_LOCK:
        return jsonify({
            "findings":      STATE["findings"][:100],
            "insights":      STATE["insights"][:20],
            "evolution_log": STATE["evolution_log"][-20:],
            "prompt_genome": STATE["prompt_genome"],
            "cycle_count":   STATE["cycle_count"],
            "distill_count": STATE["distill_count"],
            "counts": {
                "findings":   len(STATE["findings"]),
                "insights":   len(STATE["insights"]),
                "evolutions": len(STATE["evolution_log"]),
            }
        })


@app.route("/api/run_cycle", methods=["POST"])
def run_cycle():
    global _bg_thread, _current_pipeline
    data       = request.json or {}
    api_key    = data.get("apiKey", "")
    categories = data.get("categories", list(BASE_PROMPTS.keys()))
    try:
        pipeline = AgentPipeline(api_key)
        t = threading.Thread(target=pipeline.run_cycle, args=(categories,), daemon=True)
        t.start()
        return jsonify({"ok": True, "cycle": STATE["cycle_count"] + 1})
    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auto", methods=["POST"])
def auto_mode():
    global _bg_thread, _bg_stop, _current_pipeline
    data       = request.json or {}
    action     = data.get("action", "start")
    api_key    = data.get("apiKey", "")
    categories = data.get("categories", list(BASE_PROMPTS.keys()))
    interval   = int(data.get("interval_min", 45))

    if action == "stop":
        _bg_stop.set()
        sse_emit("auto_stopped", {})
        return jsonify({"ok": True, "msg": "Auto mode durduruldu"})

    if _bg_thread and _bg_thread.is_alive():
        return jsonify({"error": "Zaten çalışıyor"}), 400

    try:
        pipeline = AgentPipeline(api_key)
        _bg_stop.clear()
        _bg_thread = threading.Thread(
            target=_bg_loop, args=(pipeline, categories, interval), daemon=True
        )
        _bg_thread.start()
        sse_emit("auto_started", {"interval_min": interval})
        return jsonify({"ok": True, "interval_min": interval})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/distill", methods=["POST"])
def force_distill():
    data    = request.json or {}
    api_key = data.get("apiKey", "")
    try:
        pipeline = AgentPipeline(api_key)
        t = threading.Thread(target=pipeline.distill, daemon=True)
        t.start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear_data():
    with STATE_LOCK:
        STATE.update(default_state())
    save_state()
    sse_emit("cleared", {})
    return jsonify({"ok": True})


@app.route("/api/test", methods=["POST"])
def test_key():
    data    = request.json or {}
    api_key = data.get("apiKey", "")
    try:
        pipeline = AgentPipeline(api_key)
        text, mdl = gemini_call(pipeline.client, "Reply: NEXA_OK")
        return jsonify({"ok": True, "model": mdl, "response": text.strip()[:60]})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/stream")
def stream():
    """SSE endpoint — client subscribes here for real-time events."""
    q: list = []
    with _sub_lock:
        _subscribers.append(q)

    def generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            if q:
                payload = q.pop(0)
                yield f"data: {payload}\n\n"
            else:
                time.sleep(0.2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
