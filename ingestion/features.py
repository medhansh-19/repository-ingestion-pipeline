import re
import math
from dataclasses import dataclass

MAX_DOC_SCORE = 100

_SECTION_SCORES = {
    "installation": 15, "usage": 15, "api": 10,
    "contributing": 10, "license": 10, "faq": 5,
}
_SECTION_RE = {
    "installation": re.compile(r"\b(installation|install|setup|getting[\s_]started|quick[\s_]start)\b", re.I),
    "usage":        re.compile(r"\b(usage|examples?|how[\s_]to[\s_]use|quickstart)\b", re.I),
    "api":          re.compile(r"\b(api[\s_]?reference|api[\s_]?docs?|endpoints?)\b", re.I),
    "contributing": re.compile(r"\b(contributing|contribution[s]?)\b", re.I),
    "license":      re.compile(r"\b(licen[sc]e|mit|apache|gpl)\b", re.I),
    "faq":          re.compile(r"\b(faq|frequently[\s_]asked|troubleshoot)\b", re.I),
}

@dataclass
class DocResult:
    score: float          
    raw: int
    found: list[str]
    missing: list[str]
    breakdown: dict

def extract_tags(repo_id: str, paragraphs: list[str]) -> list[str]:
    """Extracts structural tokens from the title and semantic phrases from docs."""
    stop_words = {
        'app', 'api', 'repo', 'project', 'demo', 'test', 'tool', 'my', 'the',
        'and', 'for', 'with', 'io', 'github', 'boilerplate', 'template', 'version'
    }
    
    # 1. Extract from Title Name
    t = repo_id.split("/")[-1]
    for pat, rep in {
        r'REST(ful)?': lambda m: 'Rest' + (m.group(1) or ''),
        r'JWT': 'Jwt', r'GraphQL': 'Graphql', r'MySQL': 'Mysql',
        r'PostgreSQL': 'Postgresql', r'NoSQL': 'Nosql',
    }.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        
    t = re.sub(r'(?<![a-zA-Z0-9])c\+\+(?![a-zA-Z0-9])', 'cpp', t, flags=re.IGNORECASE)
    t = re.sub(r'(?<![a-zA-Z0-9])c#(?![a-zA-Z0-9])',    'csharp', t, flags=re.IGNORECASE)
    t = re.sub(r'\.js\b', ' js', t, flags=re.IGNORECASE)
    t = re.sub(r'\.ts\b', ' ts', t, flags=re.IGNORECASE)
    t = re.sub(r'([a-zA-Z])([0-9])', r'\1 \2', t)
    t = re.sub(r'([0-9])([a-zA-Z])', r'\1 \2', t)
    t = re.sub(r'([a-z0-9])([A-Z])',  r'\1 \2', t)
    t = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', t)
    t = re.sub(r'[-_./\\:|+=#@^&*~`?<>!]', ' ', t)
    
    tokens = re.findall(r'\b[a-zA-Z]{2,}\b', t)
    seen = set()
    tags = []
    for tok in tokens:
        lower_tok = tok.lower()
        if lower_tok not in stop_words and lower_tok not in seen:
            seen.add(lower_tok)
            tags.append(lower_tok)
    
    # 2. Extract compound semantic concepts from descriptions
    text_corpus = " ".join(paragraphs).lower()
    high_value_phrases = [
        "ai assistant", "tool calling", "local ai", "voice interaction", "voice",
        "autonomous", "agent", "llm", "rag", "vector database", "machine learning",
        "inference", "multi-agent", "frontend", "backend"
    ]
    
    for phrase in high_value_phrases:
        if phrase in text_corpus:
            title_phrase = phrase.title()
            if title_phrase.lower() not in seen:
                seen.add(title_phrase.lower())
                tags.append(title_phrase)
            
    return tags


def score_documentation(repo: dict) -> DocResult:
    """Evaluates the completeness and distribution structural depth of the README text."""
    readme_len = repo.get("readme_length", 0)
    ratio      = repo.get("readme_to_codebase_ratio", 0.0)
    text       = " ".join(repo.get("extracted_paragraphs", []))
    
    pts = 0
    bd  = {}
    if readme_len > 0:
        pts += 10; bd["readme_exists"] = 10
    if readme_len > 500:  
        pts += 10; bd["length_500"]    = 10
    if readme_len > 2000: 
        pts +=  5; bd["length_2000"]   = 5
    if ratio > 0.001:     
        pts +=  5; bd["ratio_bonus"]   = 5
        
    found, missing = [], []
    for sec, rx in _SECTION_RE.items():
        if rx.search(text):
            pts += _SECTION_SCORES[sec]
            bd[sec] = _SECTION_SCORES[sec]
            found.append(sec)
        else:
            missing.append(sec)
            
    raw = min(pts, MAX_DOC_SCORE)
    return DocResult(score=round(raw / MAX_DOC_SCORE, 4), raw=raw, found=found, missing=missing, breakdown=bd)


def activity_score(repo: dict) -> float:
    """Calculates active code maintainer energy incorporating real-time exponential half-life decay."""
    pushed_days = repo.get("pushed_days_ago", 999)
    recency  = math.exp(-pushed_days * math.log(2) / 30.0) # 30-day half life scale
    stars    = min(math.log1p(repo.get("star_count", 0)) / math.log1p(200_000), 1.0)
    contrib  = min(repo.get("mentionable_users_count", 0) / 10.0, 1.0)
    return round(0.50 * recency + 0.35 * stars + 0.15 * contrib, 4)


def trend_velocity(repo: dict) -> float:
    """Quantifies current social popularity traction across multi-stage window intervals."""
    r3  = repo.get("delta_3d",  0) / 3.0
    r7  = repo.get("delta_7d",  0) / 7.0
    r30 = repo.get("delta_30d", 0) / 30.0
    
    blend = 0.50 * r3 + 0.30 * r7 + 0.20 * r30
    vel   = min(math.log1p(blend) / math.log1p(500), 1.0)
    
    # Acceleration boost flag for sudden exponential breakouts
    if r30 > 0 and r3 > r30:
        vel = min(vel + 0.15 * min((r3 - r30) / max(r30, 1), 1.0), 1.0)
    return round(vel, 4)


def build_structured_summary(repo: dict, tags: list[str], category: str) -> str:
    """
    Standardizes variable-length documentation into a structured presentation blueprint.
    Ensures that evaluating systems maintain feature-weight normalization parity.
    """
    name = repo.get("id", "unknown/repo").split("/")[-1]
    lang = repo.get("primary_language", "Unknown") or "Unknown"
    stars = repo.get("star_count", 0)
    paras = repo.get("extracted_paragraphs", [])
    
    # Process text layout abstracts
    clean_paras = [re.sub(r"<[^>]+>", " ", p).strip() for p in paras]
    clean_paras = [re.sub(r"\s+", " ", p) for p in clean_paras if len(p) > 30]
    
    abstract = max(clean_paras, key=len) if clean_paras else "(No core architectural documentation synopsis available)"
    if len(abstract) > 300:
        abstract = abstract[:300].rsplit(" ", 1)[0] + "..."

    lines = [
        f"Repository: {name}",
        f"Category: {category}",
        f"Primary Language: {lang}",
        f"Stars: {stars:,}",
        "",
        "Core Purpose:",
        f"  {abstract}",
        "",
        "Technology Stack / Extracted Tags:",
        f"  {', '.join(tags[:12]) if tags else 'No semantic keywords flagged'}"
    ]
    return "\n".join(lines)
