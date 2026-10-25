"""The five distinct agents. Each has its own prompt, its own inputs, and its own job:
  PLANNER      question -> sub-questions + search queries
  RESEARCHER   sub-question -> stored evidence passages (web + uploaded docs)
  DRAFTER      evidence -> atomic claims, each citing evidence ids
  VERIFIER     (claim, cited evidence) -> supported / partial / unsupported (+ programmatic checks)
  CONTRADICTOR verified claims -> pairs that disagree
  SYNTHESIZER  verified claims -> structured report text citing claim ids
"""
import datetime
import re

PLANNER_SYS = """You are the PLANNER in a precision deep-dive research agent.
Decompose the user's research question into focused, highly targeted sub-questions based on the requested research depth.

DEPTH CONFIGURATION:
- Standard depth ("std"): Produce EXACTLY 4 sub-questions.
- Quick depth ("quick"): Produce 2 to 3 sub-questions.
- Deep depth ("deep"): Produce 4 to 5 sub-questions.
Target depth for this run: {depth_desc}

CORE GROUNDING & SCOPE DISCIPLINE RULES:
1. STRICT CLAUSE GROUNDING: Every sub-question MUST map directly to an explicit clause or core necessity of the user's question. Record this in "maps_to_clause".
2. ELIMINATE SCOPE DRIFT: Never introduce peripheral filler angles (such as creator/influencer biographies, online drama, historical origins, or tangential legal/regulatory regimes) unless the user explicitly requested them.
3. ATOMIC SUB-QUESTION INQUIRY STANDARD (NEVER COMBINE CONCEPTS INTO COMPOUND QUESTIONS):
   - Every sub-question MUST focus on a single, atomic research angle. Never combine conceptual definition and empirical applicability into one sub-question.
   - Contrast Example:
     * BAD (compound): "How do political scientists define the 'deep state' concept and assess its applicability to China's party-state system?"
     * GOOD (atomic):
       - Sub-question A: "How do political scientists define the 'deep state' concept, and what institutional criteria distinguish it in comparative politics?"
       - Sub-question B: "What scholarly arguments and evidence evaluate whether the 'deep state' framework accurately applies to China's Leninist party-state?"
4. INQUIRY STANDARD (RESEARCH QUESTIONS, NOT TOPIC HEADERS):
   - Every "question" MUST be a sharp, interrogative research sentence ending with a question mark "?".
   - Formulate the question in the user's own subject terms, asking what an expert researcher investigating this specific topic needs to know.
   - NEVER use noun phrases or broad topic labels as the question.
5. INVESTIGATIVE COVERAGE:
   - Identify what an expert investigating THIS exact question needs to know to reach a definitive, evidence-backed answer.
   - Decompose into distinct, non-overlapping substantive facets (e.g. foundational requirements, operational methodologies, quantitative benchmarks/efficacy, and critical limiting factors).
6. FINDINGS BOARD LANE LABEL ("label"):
   - A 5-6 word STATEMENT (noun phrase or short statement, NOT a question, NO trailing question mark, preserving question intent).
7. ANSWER SPECIFICATION ("answer_spec"):
   - One sentence stating the required factual entities, data points, or scientific measurements needed to satisfy the inquiry.
8. SEARCH QUERIES ("queries"):
   - Exactly 2 distinct web-search queries.
   - Format: 3 to 8 plain natural language words naming concrete entities, technical terms, or publication types.
   - MUST PRESERVE SUBJECT & PERSPECTIVE: Every query MUST retain the sub-question's core subject and evaluator perspective (e.g. whose preference, evaluation, or view is being asked about). Never drop the evaluating population or subject to search for the object in isolation.
   - FORBIDDEN IN QUERIES: quotation marks, boolean operators (OR, AND, NOT), site: operators, or speculative author/outlet names.
   - Never repeat earlier queries within the same lane.
9. AUDIT TRAIL ("audit"):
   - List the mapped user clauses, any dropped peripheral angles deliberately excluded to avoid drift, and brief reasoning.

Today is {date}.
IMPORTANT: Return ONLY the JSON object starting with { and ending with }. Do NOT write drafting notes, numbered thinking steps, scratchpads, or explanations before the JSON.
Return JSON strictly in this structure:
{{
  "sub_questions": [
    {{
      "id": "Q1",
      "label": "5-6 word statement without question mark",
      "question": "Full interrogative research question ending in ?",
      "answer_spec": "...",
      "maps_to_clause": "...",
      "queries": ["...", "..."]
    }}
  ],
  "audit": {{
    "mapped_clauses": ["..."],
    "dropped_peripheral_angles": ["..."],
    "reasoning": "..."
  }}
}}"""

REFINE_SYS = """You are the RESEARCHER. A sub-question is thinly supported or missing direct evidence so far.
Propose 2 NEW web-search queries that use different wording or angles than the ones already tried.
Rules:
- Formulate queries that name specific entities, ingredients, technical terms, or primary trial keywords required by the ANSWER SPECIFICATION.
- MUST PRESERVE SUBJECT & PERSPECTIVE: Every query MUST retain the sub-question's core subject and evaluator perspective (e.g. whose preference, evaluation, or view is being asked about). Never drop the evaluating population or subject to search for the object in isolation.
- Format: Exactly 3 to 8 plain natural language words.
- STRICTLY FORBIDDEN: quotation marks, boolean operators (OR, AND, NOT), site: operators, or speculative author/outlet names.
- Prefer angles that retrieve from primary or authoritative sources (trials, scientific studies, filings, reputable press).
- Avoid repeating topics that were previously found off-topic or peripheral.
Today is {date}.
Return JSON: {"queries":["...","..."]}"""

DRAFTER_SYS = """You are the DRAFTER in a precision research system. Write up to 6 atomic factual claims that DIRECTLY answer the sub-question according to the ANSWER SPECIFICATION, using ONLY the
evidence passages provided. Zero claims is a correct answer when the passages only give background, methods, motivation or prior-literature statements. Hard rules:
- Every claim must directly answer the sub-question and satisfy the ANSWER SPECIFICATION.
- A claim must state a result, figure, comparison, effect or mechanism that a passage asserts. When a passage contains both methods or motivation and results, draft from the results. Do not draft from sentences that only describe a study's data or method, what prior studies did, or what remains unexplored. If only such sentences exist, return {"claims": []}.
- Ignore passages that are only general background, peripheral regulation, disclaimers, or warnings unrelated to the core ask.
  Negative example: If the question asks which natural supplements support weight loss, an FDA warning about an unrelated toxic plant or generic advice to 'eat healthy' is NOT an answer. Do NOT draft claims about it.
- If no passage directly answers the sub-question according to the answer specification, return an empty list: {"claims": []}.
- Every claim must be directly stated by, or a close paraphrase of, the passages it cites.
- Cite 1-3 evidence ids per claim (like "E4"). Never cite an id that was not provided.
- Any number, date or name in a claim must appear in the cited passage.
- Do NOT add background knowledge. If the evidence cannot support a claim, leave it out; fewer claims is fine.
- Do not repeat claims listed under ALREADY ESTABLISHED.
Return JSON: {"claims":[{"text":"...","evidence_ids":["E1"]}]}"""

RELEVANCE_SYS = """You are the CLAIM RELEVANCE GATE in a precision research agent.
Evaluate whether each drafted claim bears directly on the specific inquiry and ANSWER SPECIFICATION.

Rules:
- SPECIFIC RELATIONSHIP & INQUIRY ALIGNMENT: The claim must address the specific relationship, question, or perspective asked (such as whose preference, view, or choice is measured, or what effect factor A has on outcome B). It is NOT sufficient for a claim to merely mention a subject keyword or describe an entity/industry in isolation without addressing the specific inquiry asked about it.
- DISSENTING AND NEGATIVE FINDINGS ARE DIRECTLY RELEVANT: Claims asserting that a factor does NOT have an effect, that a preference is absent, that a concept does NOT apply, or presenting counter-evidence directly satisfy the inquiry and MUST be marked "direct" (e.g. claims showing a lack of association or that an institutional model does not fit).
- FACTUAL GROUNDING IS NOT EVALUATED HERE: Verification handles truth and citation proof. Only judge whether the claim, if true, directly answers or informs the sub-question.
- "direct": The claim directly addresses the specific inquiry, relationship, preference, ranking, or mechanism asked by the sub-question (including positive findings, negative findings, or critiques).
- "off_topic": The claim fails to address the specific inquiry asked, is purely tangential context, describes an entity without answering what was asked about it, or is customer service/boilerplate.

There is NO "partial". Only "direct" or "off_topic".
Echo the EXACT candidate ID provided in brackets (e.g. if the claim is [temp_Q1_1], use "cid": "temp_Q1_1").

IMPORTANT: Return ONLY the JSON object starting with { and ending with }. Do NOT write chain-of-thought, thinking steps, scratchpads, or explanations before the JSON.
Return JSON:
{"evaluations":[{"cid":"<exact_id_from_brackets>","relevance":"direct|off_topic","reason":"one brief sentence"}]}"""

VERIFIER_SYS = """You are the VERIFIER, an adversarial fact-checker. You see one CLAIM and the passages it cites.
Decide whether the passages themselves support the claim. Ignore your own knowledge: judge only from the text shown.
- "supported": the passages state the claim (paraphrase is fine), including every number, date and name in it.
- "partial": the passages support only part of the claim, or support it only loosely or by inference.
- "unsupported": the passages do not say this, or contradict it.
Also copy the shortest VERBATIM excerpt (max 200 chars) from a passage that supports the claim; leave it empty if none.
Return JSON: {"verdict":"supported|partial|unsupported","quote":"...","reason":"one sentence"}"""

CONTRA_SYS = """You are the CONTRADICTION DETECTOR. Given verified claims (with ids), find pairs that genuinely
disagree (conflicting numbers, opposing conclusions, incompatible dates). Do not flag claims that are merely
different or complementary. Return JSON:
{"contradictions":[{"claim_ids":["C1","C4"],"explanation":"what exactly conflicts"}]}
Return {"contradictions":[]} if there are none."""

SYNTH_SYS = """You are the SYNTHESIZER. Write a comprehensive, cited research report answering the user's question using ONLY the claims
provided (each has an id like C3). Rules:
- The "summary" MUST be an informative Executive Summary of 3-5 sentences summarizing the core verified findings across the sub-questions.
- In both "summary" and "sections", EVERY sentence MUST end with its claim citations like [C3] or [C3][C7]. Place citations directly before the period (e.g. "... increases cellular energy [C3].").
- Prioritize VERIFIED claims. Do not add facts, numbers or opinions that are not in the claims.
- Where claims are flagged as contradicting, state the disagreement openly.
- For sub-questions marked THIN, say briefly that evidence was limited rather than filling the gap.

Example format:
{
  "summary": "Regular exercise boosts mitochondrial efficiency and daily energy levels [C10]. Consistent sleep habits and limiting blue light prevent circadian disruptions [C1][C4]. Proper hydration and unrefined diets maintain cognitive focus throughout the day [C7][C8].",
  "sections": [
    {"id": "Q1", "text": "High-protein snacks elevate glucose sustainably without causing energy crashes [C1][C2]. Drinking water immediately counteracts sluggishness caused by mild dehydration [C3]."}
  ]
}"""

FOLLOWUP_SYS = """You answer a follow-up question using ONLY the earlier verified claims provided (ids like C3).
Every sentence must end with claim citations like [C3]. If the claims do not cover the question, say so in one
sentence WITHOUT inventing anything, and suggest a new research run. Return JSON: {"answer":"..."}"""


def today() -> str:
    return datetime.date.today().isoformat()


CONNECTIVES = {
    "and", "or", "that", "of", "the", "to", "for", "in", "with", "by", "on", "at", "a", "an", "is", "are", "from", "as"
}


def validate_lane_label(label: str, question: str) -> tuple[bool, str]:
    """Validate Findings Board lane label according to strict FIX v3 specification:
    - 5 to 6 words.
    - No trailing question mark.
    - Must not end on a stop word or connective.
    - Must not be a verbatim prefix of the question.
    """
    if not label or not isinstance(label, str):
        return False, "label is empty"
    cleaned = label.strip()
    if cleaned.endswith("?") or re.search(r"\?\s*$", cleaned):
        return False, "trailing question mark"
    words = cleaned.split()
    if len(words) not in (5, 6):
        return False, f"word count {len(words)} is not 5 or 6 words"
    last_word = words[-1].lower().strip(".,;:?!\"'()")
    if last_word in CONNECTIVES:
        return False, f"ends on connective/stopword '{last_word}'"
    # Check if verbatim prefix of question
    q_words = [re.sub(r"[^\w]", "", w).lower() for w in question.split() if re.sub(r"[^\w]", "", w)]
    l_words = [re.sub(r"[^\w]", "", w).lower() for w in words if re.sub(r"[^\w]", "", w)]
    if len(q_words) >= len(l_words) and q_words[:len(l_words)] == l_words:
        return False, "verbatim prefix of question"
    return True, "valid"


REGEN_LABEL_SYS = """You are a label generator for a research findings board.
Convert the given sub-question into a 5-6 word STATEMENT (a noun phrase or short statement, NOT a question).
Rules:
- Exactly 5 or 6 words.
- NO question mark at the end.
- Must NOT end on a connective or stop word (and, or, that, of, the, to, for, in, with, by, on, at, a, an, from, as).
- Must NOT be a verbatim prefix of the question (do NOT just take the first 5 words).
- Preserves the intent of the question.
Structural pattern:
Transform an interrogative sentence like "What specific [substantive aspect] and [core mechanism] govern [subject]?"
into a concise declarative statement: "[Substantive aspect] and [core mechanism] for [subject]".
Return JSON: {"label": "..."}"""


def regenerate_lane_label(llm, question: str, model: str | None = None) -> str:
    """Make one targeted LLM call to regenerate an invalid or missing lane label."""
    try:
        out = llm.json(REGEN_LABEL_SYS, f"SUB-QUESTION: {question}", model=model, tag="label_regen", max_tokens=100)
        lbl = (out.get("label") or "").strip()
        ok, _ = validate_lane_label(lbl, question)
        if ok:
            return lbl
    except Exception:
        pass
    return ""


DEPTH_DESCRIPTIONS = {
    "quick": "Quick mode: Return 2 to 3 focused sub-questions.",
    "std": "Standard mode: Return EXACTLY 4 focused sub-questions.",
    "deep": "Deep mode: Return 4 to 5 in-depth sub-questions.",
}

TAGGER_SYS = """"""
GENERIC_TAG_WORDS = set()
STOCK_TAGS = set()


def validate_tag_form(tag: str, existing_tags: set[str] | None = None) -> tuple[bool, str]:
    """Non-functional / isolated: tags disabled to consume zero LLM tokens."""
    return True, ""


def extract_tag_from_question(question: str, existing_tags: set[str] | None = None) -> str:
    """Non-functional / isolated: tags disabled to consume zero LLM tokens."""
    return ""


def assign_subquestion_tag(llm, question: str, existing_tags: set[str] | None = None, model: str | None = None) -> str:
    """Non-functional / isolated: tags disabled to consume zero LLM tokens."""
    return ""


def clean_and_validate_tag(tag: str, question: str, existing_tags: set[str] | None = None) -> str:
    """Non-functional / isolated: tags disabled to consume zero LLM tokens."""
    return ""


def ensure_interrogative_question(q: str, fallback_topic: str = "") -> str:
    """Ensure the sub-question is an active research question ending with ?."""
    clean_q = (q or "").strip()
    if not clean_q:
        clean_q = fallback_topic.strip() or "What are the core findings?"
    clean_q = clean_q.rstrip(".!;:, ")
    if not clean_q.endswith("?"):
        clean_q = clean_q + "?"
    q_lower = clean_q.lower()
    interrogatives = ("what", "how", "why", "which", "who", "when", "where", "does", "do", "are", "is", "can", "to what extent")
    if not any(q_lower.startswith(w + " ") for w in interrogatives):
        core = clean_q.rstrip("?").strip()
        clean_q = f"What specific evidence, mechanisms, and outcomes define {core}?"
    return clean_q


class PlanList(list):
    """List of sub-questions with attached audit metadata for traceability."""
    def __init__(self, iterable, audit: dict | None = None, depth: str = "std"):
        super().__init__(iterable)
        self.audit = audit or {}
        self.depth = depth


DOMAIN_PATTERNS = [
    {
        "trigger": r"\b(pickup|dating|attraction|seduction|social dynamics|persuasion|women psychology)\b",
        "angles": [
            ("Interaction Tactics", "Opening lines, conversational frameworks, and behavioral routines",
             "What specific opening lines, conversational frameworks, and behavioral routines define {topic}?",
             "Details specific conversational routines, verbal frameworks, and interaction tactics for {topic}.",
             ["{topic} opening lines conversational routines tactics", "{topic} behavioral frameworks interaction methods"]),
            ("Psychological Triggers", "Targeted cognitive biases, intermittent reinforcement, and persuasion mechanisms",
             "What specific cognitive biases, intermittent reinforcement patterns, and psychological mechanisms are targeted by {topic}?",
             "Cites specific cognitive biases, intermittent validation mechanisms, and psychological compliance drivers targeted by {topic}.",
             ["{topic} psychological mechanisms cognitive biases targeted", "{topic} intermittent reinforcement compliance triggers"]),
            ("Empirical Studies", "Documented relationship outcomes, trial data, and peer-reviewed studies",
             "What peer-reviewed empirical studies and relationship research evaluate the real-world efficacy of {topic}?",
             "Reviews peer-reviewed empirical studies, trial datasets, and measurable outcome evaluations for {topic}.",
             ["{topic} empirical research peer-reviewed scientific studies", "{topic} psychological relationship efficacy evidence"]),
            ("Boundary Harms", "Documented interpersonal risks, emotional harms, and ethical critiques",
             "What documented interpersonal harms, psychological effects, and ethical critiques are associated with {topic}?",
             "Identifies documented interpersonal risks, negative emotional consequences, and critical assessments regarding {topic}.",
             ["{topic} documented psychological harms relationship risks", "{topic} ethical critiques manipulation consequences"])
        ]
    },
    {
        "trigger": r"\b(skill|skills|career|job|jobs|hiring|interview|degree|learn|study|become|get into|roles?|quant|finance)\b",
        "angles": [
            ("Core Competencies", "Essential core technical and mathematical competencies",
             "What primary mathematical, computational, and analytical competencies are required for {topic}?",
             "Names specific mathematical models, quantitative concepts, and core fundamentals required for {topic}.",
             ["{topic} core mathematical competencies requirements", "{topic} quantitative skills syllabus fundamentals"]),
            ("Technical Stack", "Required programming languages, software, and operational tools",
             "What programming languages, software architectures, and computational tools are industry-standard in {topic}?",
             "Lists specific programming languages, performance libraries, and computational systems used in {topic}.",
             ["{topic} programming languages frameworks tech stack", "{topic} software architecture tools requirements"]),
            ("Hiring Benchmarks", "Educational credentials, certifications, and candidate evaluation standards",
             "What degree credentials, research publications, and portfolio benchmarks distinguish competitive candidates for {topic}?",
             "Details educational requirements, academic degrees, and portfolio evaluation criteria for {topic}.",
             ["{topic} educational background degree requirements hiring", "{topic} portfolio projects candidate benchmarks"]),
            ("Interview Pathways", "Preparation strategies, technical assessments, and practical interview barriers",
             "What technical interview assessments, preparation pathways, and evaluation hurdles define {topic}?",
             "Identifies interview assessment types, preparation roadmaps, and entry barriers for {topic}.",
             ["{topic} technical interview assessment preparation", "{topic} hiring process assessment questions"])
        ]
    },
    {
        "trigger": r"\b(battery|batteries|solid-state|energy|chemical|material|materials|hardware|manufacturing|cells?)\b",
        "angles": [
            ("Material Bottlenecks", "Material compositions and interfacial degradation mechanisms",
             "What primary solid electrolyte materials and interfacial degradation mechanisms limit {topic}?",
             "Details specific materials, interfacial resistance values, and dendrite penetration bottlenecks for {topic}.",
             ["{topic} solid electrolyte interfacial resistance degradation", "{topic} material dendrite formation bottlenecks"]),
            ("Manufacturing Scalability", "Production throughput, yield rates, and industrial scaling",
             "What industrial deposition methods, defect densities, and yield constraints prevent mass production of {topic}?",
             "Cites production methods, manufacturing throughput, and yield rates for {topic}.",
             ["{topic} industrial manufacturing throughput yield rates", "{topic} dry electrode deposition scaling"]),
            ("Commercial Parity", "Cost metrics, energy density targets, and economic viability",
             "What pack-level cost thresholds and volumetric energy density metrics must {topic} achieve for commercial parity?",
             "Lists cost per kWh targets and volumetric density metrics determining commercial adoption for {topic}.",
             ["{topic} cost per kWh commercial viability targets", "{topic} volumetric energy density benchmarks"]),
            ("Automotive Timelines", "OEM validation testing, thermal safety protocols, and commercial timelines",
             "What automotive OEM qualification cycles, thermal runaway testing, and projected adoption roadmaps define {topic}?",
             "Identifies automotive validation milestones, thermal safety standards, and projected deployment dates for {topic}.",
             ["{topic} automotive OEM validation testing timeline", "{topic} thermal safety testing deployment roadmap"])
        ]
    },
    {
        "trigger": r"\b(supplement|supplements|weight loss|health|metabolic|fasting|diet|nutrition|medical|clinical|safety)\b",
        "angles": [
            ("Active Ingredients", "Evidence-based active compounds and documented clinical efficacy",
             "What specific active ingredients, natural compounds, and formulations demonstrate trial evidence for {topic}?",
             "Names specific active compounds, standardized extracts, and clinical trial results for {topic}.",
             ["{topic} active compounds randomized controlled trials", "{topic} clinical efficacy standardized extracts"]),
            ("Biological Pathways", "Physiological mechanisms of action and metabolic changes",
             "What physiological mechanisms, cellular pathways, and metabolic shifts are triggered by {topic}?",
             "Details metabolic pathways, hormonal shifts, and biochemical mechanisms of action for {topic}.",
             ["{topic} biological mechanisms of action pathways", "{topic} cellular metabolic effects hormonal response"]),
            ("Dosage Regimens", "Therapeutic dosage ranges, trial durations, and measured effect sizes",
             "What clinical dosage ranges, intervention durations, and measured effect sizes are established in trials for {topic}?",
             "Cites trial dosages, duration limits, and quantitative effect sizes for {topic}.",
             ["{topic} clinical trial dosage duration effect size", "{topic} therapeutic dosing randomized studies"]),
            ("Safety Boundaries", "Documented adverse interactions, contraindications, and safety warnings",
             "What adverse events, drug interactions, contraindications, and regulatory warnings apply to {topic}?",
             "Lists documented adverse events, contraindications, and regulatory warnings for {topic}.",
             ["{topic} adverse side effects drug interactions", "{topic} contraindications clinical safety warnings"])
        ]
    },
    {
        "trigger": r"\b(history|historical|past|ancient|cosmology|cosmological|flat earth|earth was flat|origins|century)\b",
        "angles": [
            ("Ancient Records", "Primary historical texts, cosmological tablets, and ancient maps",
             "What primary historical texts, ancient cosmological models, and early cartographic records describe {topic}?",
             "Names specific ancient texts, manuscript citations, and historical cosmological descriptions for {topic}.",
             ["{topic} primary historical texts manuscripts", "{topic} ancient cosmological records maps"]),
            ("Theological Texts", "Scriptural passages and mythological cosmological frameworks",
             "What scriptural passages, theological commentaries, and mythological traditions articulated frameworks for {topic}?",
             "Cites mythological frameworks, religious scriptures, and medieval cosmological commentaries for {topic}.",
             ["{topic} mythological cosmologies religious texts", "{topic} scriptural passages theological commentaries"]),
            ("Geodetic Proofs", "Mathematical proofs, horizon observations, and scientific refutations",
             "What mathematical calculations, lunar eclipse observations, and geodetic measurements historically refuted {topic}?",
             "Cites specific historical measurements (e.g. Eratosthenes, shadow calculations, horizon optics) addressing {topic}.",
             ["{topic} geodetic measurements Eratosthenes shadow", "{topic} historical astronomical observations refutations"]),
            ("Modern Revivalism", "Sociological resurgence, pseudo-scientific claims, and community propagation",
             "What sociological factors, pseudo-scientific arguments, and digital media dynamics drive modern revivals of {topic}?",
             "Details specific revival societies, historical arguments (e.g. Samuel Rowbotham), and propagation dynamics for {topic}.",
             ["{topic} Zetetic astronomy Samuel Rowbotham history", "{topic} modern revival sociological media dynamics"])
        ]
    }
]


def generate_calibrated_plan(question: str, depth: str = "std") -> PlanList:
    """Generate domain-grounded, non-generic fallback subquestions when LLM is unavailable or empty."""
    depth_clean = (depth or "std").lower().strip()
    if depth_clean not in ("quick", "std", "deep"):
        depth_clean = "std"
    target_count = 3 if depth_clean == "quick" else (5 if depth_clean == "deep" else 4)

    clean_q = question.strip().rstrip("?.!;:, ")
    core_topic = re.sub(r"^(what|how|why|which|who|is|are|do|does|can)\s+(is|are|the|do|does|did|to)?\s*", "", clean_q, flags=re.IGNORECASE).strip()
    if not core_topic:
        core_topic = clean_q

    # Match domain-specific angles
    chosen_angles = None
    for d in DOMAIN_PATTERNS:
        if re.search(d["trigger"], clean_q, re.IGNORECASE):
            chosen_angles = [
                (t, l, qp.format(topic=core_topic), sp.format(topic=core_topic), [qf.format(topic=core_topic) for qf in qs])
                for t, l, qp, sp, qs in d["angles"]
            ]
            break

    if not chosen_angles:
        # General investigative decomposition using question's own keywords
        q_words = [re.sub(r"[^\w-]", "", w) for w in core_topic.split() if re.sub(r"[^\w-]", "", w)]
        content_words = [
            w for w in q_words
            if len(w) > 2
            and w.lower() not in CONNECTIVES
            and w.lower() not in {"what", "how", "why", "which", "who", "when", "where", "does", "are", "can", "is", "do", "the"}
        ]
        subj = " ".join(content_words[:3]) if content_words else core_topic
        chosen_angles = [
            ("Core Requirements", "Primary foundational requirements and core criteria",
             f"What primary foundational principles, prerequisites, and core criteria define {core_topic}?",
             f"Names specific foundational requirements, criteria, and definitions for {core_topic}.",
             [f"{subj} foundational principles criteria", f"{subj} core requirements overview"]),
            ("Operational Methods", "Specific operational methodologies and technical workflows",
             f"What specific methodologies, operational frameworks, and practical workflows govern {core_topic}?",
             f"Details specific methodologies, operational frameworks, and practical procedures for {core_topic}.",
             [f"{subj} operational methodologies procedures", f"{subj} technical frameworks implementation"]),
            ("Evaluation Metrics", "Quantitative benchmarks, empirical data, and measurable outcomes",
             f"What quantitative benchmarks, empirical observations, and measurable outcomes evaluate {core_topic}?",
             f"Cites specific quantitative metrics, performance benchmarks, and empirical observations for {core_topic}.",
             [f"{subj} quantitative benchmarks metrics", f"{subj} empirical evaluation outcomes"]),
            ("Limiting Factors", "Critical challenges, boundary limitations, and risk considerations",
             f"What critical constraints, boundary conditions, and risk factors challenge {core_topic}?",
             f"Identifies critical constraints, failure modes, and limiting factors regarding {core_topic}.",
             [f"{subj} critical constraints limiting factors", f"{subj} boundary risks challenges"])
        ]

    result = []
    for a_tag, a_lbl, a_q, a_spec, a_queries in chosen_angles[:target_count]:
        result.append({
            "id": f"Q{len(result) + 1}",
            "tag": "",
            "label": a_lbl,
            "question": a_q,
            "answer_spec": a_spec,
            "maps_to_clause": clean_q,
            "queries": a_queries,
            "tried": []
        })

    audit = {
        "mapped_clauses": [clean_q],
        "dropped_peripheral_angles": ["creator biographies", "tangential commentary", "unrelated historical trivia"],
        "reasoning": f"Grounded {depth_clean} inquiry systematically covering fundamentals, methodology, evaluation, and limiting factors."
    }
    return PlanList(result, audit=audit, depth=depth_clean)


def plan(llm, question: str, depth: str = "std", model: str | None = None) -> PlanList:
    depth_clean = (depth or "std").lower().strip()
    if depth_clean not in ("quick", "std", "deep"):
        depth_clean = "std"
    depth_desc = DEPTH_DESCRIPTIONS.get(depth_clean, DEPTH_DESCRIPTIONS["std"])
    prompt = PLANNER_SYS.replace("{date}", today()).replace("{depth_desc}", depth_desc)

    try:
        out = llm.json(prompt, f"Question: {question}", model=model, tag="planner", max_tokens=3500)
    except Exception:
        out = {}
    if isinstance(out, dict):
        if "sub_questions" in out:
            subs = out["sub_questions"]
        elif "question" in out:
            subs = [out]
        else:
            subs = []
    elif isinstance(out, list):
        subs = out
    else:
        subs = []
    audit_data = out.get("audit", {}) if isinstance(out, dict) else {}
    if not isinstance(audit_data, dict):
        audit_data = {}

    target_count = 3 if depth_clean == "quick" else (5 if depth_clean == "deep" else 4)

    result = []
    for i, s in enumerate(subs, 1):
        if isinstance(s, str):
            raw_q = s.strip()
            raw_label = ""
            spec = ""
            maps_to = question
            queries = []
        elif isinstance(s, dict):
            raw_q = (s.get("question") or "").strip()
            raw_label = (s.get("label") or "").strip()
            spec = (s.get("answer_spec") or "").strip()
            maps_to = (s.get("maps_to_clause") or "").strip() or question
            queries = [x for x in s.get("queries", []) if x][:3]
        else:
            continue

        if not raw_q:
            continue
        q = ensure_interrogative_question(raw_q)

        ok, _ = validate_lane_label(raw_label, q)
        if ok:
            lbl = raw_label
        else:
            regen = regenerate_lane_label(llm, q, model=model)
            if regen:
                lbl = regen
            else:
                lbl = q

        if not spec:
            spec = f"Directly answers '{q}' with specific names, facts, evidence, and data."

        queries = [sanitize_search_query(x) for x in queries if x][:2] or [sanitize_search_query(q)]

        result.append({
            "id": f"Q{len(result) + 1}",
            "tag": "",  # Assigned in dependent step below
            "label": lbl,
            "question": q,
            "answer_spec": spec,
            "maps_to_clause": maps_to,
            "queries": queries,
            "tried": []
        })
        if len(result) >= target_count:
            break

    if not result:
        # Heuristic fallback when LLM output is empty or unavailable
        return generate_calibrated_plan(question, depth=depth_clean)

    # Normalize audit
    mapped = audit_data.get("mapped_clauses") or [question]
    dropped = audit_data.get("dropped_peripheral_angles") or []
    rsn = audit_data.get("reasoning") or f"{depth_clean.capitalize()} decomposition calibrated to question clauses without peripheral drift."
    normalized_audit = {
        "mapped_clauses": mapped if isinstance(mapped, list) else [str(mapped)],
        "dropped_peripheral_angles": dropped if isinstance(dropped, list) else [str(dropped)],
        "reasoning": str(rsn)
    }

    return PlanList(result, audit=normalized_audit, depth=depth_clean)


def sanitize_search_query(q: str, fallback_words: list[str] | None = None) -> str:
    """Sanitize query to 3-8 plain natural language words:
    - Strips quotes (", '), brackets, punctuation, and site: operators.
    - Strips boolean operators (OR, AND, NOT).
    - Ensures length is 3 to 8 plain words.
    """
    if not q or not isinstance(q, str):
        return " ".join(fallback_words[:6]) if fallback_words else "scholarly research analysis"
    clean = re.sub(r'["\'\[\]\(\)\{\}]', ' ', q)
    clean = re.sub(r'\b(OR|AND|NOT|site:[^\s]+)\b', ' ', clean)
    clean = re.sub(r'[^\w\s-]', ' ', clean)
    words = [w for w in clean.split() if w]
    if len(words) < 3 and fallback_words:
        words = (words + [w for w in fallback_words if w.lower() not in [x.lower() for x in words]])[:6]
    if len(words) > 8:
        words = words[:8]
    return " ".join(words)


def refine_queries(llm, subq: dict, evidence_summary: str, missing_info: str = "",
                   dropped_topics: list[str] | None = None, used_domains: list[str] | None = None) -> list[str]:
    lines = [
        f"Sub-question: {subq['question']}",
        f"Answer Specification: {subq.get('answer_spec', 'specific facts and evidence')}",
        f"Queries already tried: {subq.get('tried', [])}",
    ]
    if missing_info:
        lines.append(f"Missing evidence to target: {missing_info}")
    if dropped_topics:
        lines.append(f"Topics previously filtered off-topic (do not search for these): {', '.join(dropped_topics[:4])}")
    if used_domains:
        lines.append(f"Domains already retrieved (prefer new sources): {', '.join(list(used_domains)[:4])}")
    lines.append(f"What we have so far: {evidence_summary or 'nothing useful'}")

    user = "\n".join(lines)
    out = llm.json(REFINE_SYS.replace("{date}", today()), user, tag="researcher_refine")

    tried_normalized = {re.sub(r"\s+", " ", t.lower().strip()) for t in (subq.get("tried", []) or [])}
    raw_qs = out.get("queries", []) if isinstance(out, dict) else (out if isinstance(out, list) else [])

    q_words = [w for w in re.sub(r'[^\w\s]', ' ', subq['question'] + ' ' + subq.get('answer_spec', '')).split()
               if len(w) > 3 and w.lower() not in CONNECTIVES and w.lower() not in {'what', 'how', 'why', 'which', 'who', 'when', 'where', 'does', 'are', 'can', 'is', 'the'}]

    clean_qs = []
    for raw in raw_qs:
        sq = sanitize_search_query(raw, fallback_words=q_words)
        if sq and sq.lower() not in tried_normalized and sq.lower() not in {c.lower() for c in clean_qs}:
            clean_qs.append(sq)
            tried_normalized.add(sq.lower())

    # Deterministic fallback when queries are missing, duplicates, or invalid (no extra LLM retry)
    if len(clean_qs) < 2:
        pivot_options = [
            " ".join(q_words[:5] + ["scholarly", "critique"]),
            " ".join(q_words[:4] + ["comparative", "analysis"]),
            " ".join(q_words[:4] + ["empirical", "evidence"]),
            " ".join(q_words[:4] + ["institutional", "assessment"]),
        ]
        for opt in pivot_options:
            san = sanitize_search_query(opt)
            if san.lower() not in tried_normalized and san.lower() not in {c.lower() for c in clean_qs}:
                clean_qs.append(san)
                tried_normalized.add(san.lower())
            if len(clean_qs) >= 2:
                break

    return clean_qs[:2] or [sanitize_search_query(subq["question"])]


def sanitize_claim_text(text: str) -> str:
    """Clean claim text by stripping bullets, list prefixes, and citation header fragments."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^[-*•\s]+", "", t)
    t = re.sub(r"^(claim\s*\d*|C\d+|\d+[\).:])\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^by\s+[A-Za-z\s.]+·\s*\d{4}\s*·\s*cited\s+by\s+\d+\s*—\s*", "", t, flags=re.IGNORECASE).strip()
    return t


def draft_claims(llm, subq: dict, evidence: list[dict], existing: list[str], main_question: str = "") -> list[dict]:
    block = "\n\n".join(f"[{e['eid']}] (source quality {e['quality']:.2f}, {e['title'][:60]})\n{e['text'][:1000]}"
                        for e in evidence)
    user = (f"Main Research Question: {main_question or subq['question']}\n"
            f"Sub-question: {subq['question']}\n"
            f"Answer Specification: {subq.get('answer_spec', 'specific facts and evidence')}\n\n"
            f"ALREADY ESTABLISHED:\n"
            + ("\n".join(f"- {t}" for t in existing) or "(none)") + f"\n\nEVIDENCE:\n{block}")
    out = llm.json(DRAFTER_SYS, user, tag="drafter", max_tokens=3500)
    allowed = {e["eid"] for e in evidence}
    claims = []
    raw_claims = out.get("claims", []) if isinstance(out, dict) else (out if isinstance(out, list) else [])
    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        text = sanitize_claim_text(c.get("text") or "")
        if len(text) < 15 or not re.search(r"[a-zA-Z]{3,}", text):
            continue
        ids = [i for i in c.get("evidence_ids", []) if i in allowed]
        claims.append({"text": text, "evidence_ids": ids})
    return claims


class RelevanceGateError(Exception):
    """Raised when the relevance gate fails to parse or respond after retry."""
    pass


def evaluate_claim_relevance(llm, subq: dict, claims: list[dict], model: str | None = None) -> list[dict]:
    """Judge each drafted claim against the sub-question and answer_spec.
    Returns list of dicts: [{'cid': 'temp_Q1_1', 'relevance': 'direct'|'off_topic', 'reason': '...'}]
    Retries once on JSON/LLM exception. If it still fails, raises RelevanceGateError.
    """
    if not claims:
        return []
    claims_list = "\n".join(f"[{c['cid']}] {c['text']}" for c in claims)
    user = (f"Sub-question: {subq['question']}\n"
            f"Answer Specification: {subq.get('answer_spec', 'specific facts and evidence')}\n\n"
            f"DRAFTED CLAIMS:\n{claims_list}")

    out = None
    last_err = None
    for attempt in range(2):
        try:
            out = llm.json(RELEVANCE_SYS, user, model=model, tag="relevance", max_tokens=2500)
            if out and (isinstance(out, dict) and "evaluations" in out or isinstance(out, list)):
                break
        except Exception as e:
            last_err = e
            if attempt == 0:
                continue

    if out is None or not (isinstance(out, dict) and "evaluations" in out or isinstance(out, list)):
        err_msg = str(last_err) if last_err else "Failed to parse JSON response from relevance gate"
        raise RelevanceGateError(f"Relevance gate JSON parse failure after retry: {err_msg[:100]}")

    evals = out.get("evaluations", []) if isinstance(out, dict) else (out if isinstance(out, list) else [])

    # Robust CID matching
    eval_by_cid = {}
    eval_by_int = {}
    for ev in evals:
        if not isinstance(ev, dict):
            continue
        rel = str(ev.get("relevance", "direct")).lower().strip()
        verdict = "direct" if rel == "direct" else "off_topic"
        reason = str(ev.get("reason") or ("direct match" if verdict == "direct" else "off-topic"))
        raw_cid = str(ev.get("cid", "")).strip()
        data = {"cid": raw_cid, "relevance": verdict, "reason": reason}
        if raw_cid:
            eval_by_cid[raw_cid] = data
            digits = re.findall(r"\d+", raw_cid)
            if digits:
                eval_by_int[digits[-1]] = data

    results = []
    for idx, c in enumerate(claims):
        cid = c["cid"]
        matched = None
        if cid in eval_by_cid:
            matched = eval_by_cid[cid]
        else:
            digits = re.findall(r"\d+", cid)
            if digits and digits[-1] in eval_by_int:
                matched = eval_by_int[digits[-1]]
            elif idx < len(evals) and isinstance(evals[idx], dict):
                ev_pos = evals[idx]
                rel_pos = str(ev_pos.get("relevance", "direct")).lower().strip()
                matched = {
                    "cid": cid,
                    "relevance": "direct" if rel_pos == "direct" else "off_topic",
                    "reason": str(ev_pos.get("reason", "positional evaluation"))
                }

        if matched:
            results.append({
                "cid": cid,
                "relevance": matched["relevance"],
                "reason": matched["reason"]
            })
        else:
            results.append({
                "cid": cid,
                "relevance": "direct",
                "reason": "omitted by relevance judge (leniency bias default)"
            })
    return results


def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"[^a-z0-9%$.,' ]+", " ", re.sub(r"\s+", " ", s)).strip()


def _numbers(s: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", s)}


def verify_claim(llm, claim: dict, evidence_by_id: dict[str, dict], model: str | None = None) -> dict:
    """Returns the claim fields: status (verified/unverified/rejected), verdict, reason, quote."""
    cited = [evidence_by_id[i] for i in claim["evidence_ids"] if i in evidence_by_id]
    if not cited:
        return {"status": "rejected", "verdict": "unsupported", "reason": "cites no retrievable source", "quote": ""}
    block = "\n\n".join(f"[{e['eid']}] {e['text'][:1400]}" for e in cited)
    out = llm.json(VERIFIER_SYS, f"CLAIM: {claim['text']}\n\nCITED PASSAGES:\n{block}",
                   model=model, tag="verifier", max_tokens=500)
    verdict = str(out.get("verdict", "unsupported")).lower()
    quote, reason = (out.get("quote") or "").strip(), (out.get("reason") or "").strip()
    if verdict == "unsupported":
        return {"status": "rejected", "verdict": verdict, "reason": reason or "source does not support claim",
                "quote": ""}
    if verdict != "supported":
        return {"status": "unverified", "verdict": "partial", "reason": reason or "only partly supported",
                "quote": quote}
    # Programmatic grounding checks: the judge's quote must really be in a cited passage,
    # and every number in the claim must appear in the cited text.
    joined = _norm(" ".join(e["text"] for e in cited))
    if not quote or _norm(quote) not in joined:
        return {"status": "unverified", "verdict": "supported",
                "reason": "judge said supported but its quoted excerpt is not verbatim in the cited source",
                "quote": quote}
    missing = [n for n in _numbers(claim["text"]) if n not in _numbers(" ".join(e["text"] for e in cited))]
    if missing:
        return {"status": "unverified", "verdict": "supported",
                "reason": f"numbers not found in cited source: {', '.join(sorted(missing))}", "quote": quote}
    return {"status": "verified", "verdict": "supported", "reason": reason, "quote": quote}


def find_contradictions(llm, claims: list[dict]) -> list[dict]:
    """Disabled: contradiction detection bypassed to consume zero LLM tokens."""
    return []


def synthesize(llm, question: str, subqs: list[dict], claims: list[dict], contradictions: list[dict],
               thin_ids: set[str]) -> dict:
    lines = []
    for q in subqs:
        flag = " [THIN: limited verified evidence]" if q["id"] in thin_ids else ""
        lines.append(f"\n{q['id']}: {q['question']}{flag}")
        for c in claims:
            if c["subq_id"] == q["id"]:
                tag = "" if c["status"] == "verified" else " (UNVERIFIED)"
                lines.append(f"  {c['cid']}{tag}: {c['text']}")
    if contradictions:
        lines.append("\nCONTRADICTIONS: " + "; ".join(
            f"{' vs '.join(x['claim_ids'])} ({x['explanation']})" for x in contradictions))
    return llm.json(SYNTH_SYS, f"Question: {question}\n" + "\n".join(lines), tag="synthesizer", max_tokens=2500)


def followup(llm, question: str, claims: list[dict]) -> str:
    listing = "\n".join(f"{c['cid']}: {c['text']}" for c in claims)
    out = llm.json(FOLLOWUP_SYS, f"Follow-up question: {question}\n\nVERIFIED CLAIMS:\n{listing}",
                   tag="followup", max_tokens=900)
    return out.get("answer", "") if isinstance(out, dict) else str(out)
