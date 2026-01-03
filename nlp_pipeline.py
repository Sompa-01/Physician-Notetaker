"""
nlp_pipeline.py - Memory Optimized Version
Uses lazy loading to reduce memory footprint
"""

import re
from typing import List, Dict, Any, Optional

# Global variables for lazy loading
_nlp = None
_kw_model = None
_sentiment_pipeline = None

# Medical phrase lists
MEDICAL_SYMPTOMS = [
    "neck pain", "back pain", "head impact", "headache", "stiffness",
    "backache", "pain", "nausea", "dizziness", "anxiety",
]

MEDICAL_DIAGNOSES = [
    "whiplash", "whiplash injury", "lower back strain", "concussion", "sprain",
]

MEDICAL_TREATMENTS = [
    "physiotherapy", "physiotherapy sessions", "painkillers",
    "analgesics", "rest", "ice", "heat therapy",
]

PROGNOSIS_TERMS = [
    "full recovery", "recovery expected", "no long-term damage",
    "no long term damage", "improving",
]


def get_nlp():
    """Lazy load spaCy model"""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_kw_model():
    """Lazy load KeyBERT model"""
    global _kw_model
    if _kw_model is None:
        try:
            from keybert import KeyBERT
            _kw_model = KeyBERT(model="distilbert-base-nli-mean-tokens")
        except Exception:
            _kw_model = None
    return _kw_model


def get_sentiment_pipeline():
    """Lazy load sentiment pipeline"""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            _sentiment_pipeline = pipeline("sentiment-analysis")
        except Exception:
            _sentiment_pipeline = None
    return _sentiment_pipeline


def _make_phrase_matcher(nlp, term_list: List[str], label: str):
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(text) for text in term_list]
    matcher.add(label, patterns)
    return matcher


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract medical entities using spaCy and pattern matching"""
    nlp = get_nlp()
    doc = nlp(text)
    
    symptoms = set()
    diagnoses = set()
    treatments = set()
    prognosis = set()

    # Create matchers on demand
    sym_matcher = _make_phrase_matcher(nlp, MEDICAL_SYMPTOMS, "SYMPTOM")
    diag_matcher = _make_phrase_matcher(nlp, MEDICAL_DIAGNOSES, "DIAGNOSIS")
    treat_matcher = _make_phrase_matcher(nlp, MEDICAL_TREATMENTS, "TREATMENT")
    prog_matcher = _make_phrase_matcher(nlp, PROGNOSIS_TERMS, "PROGNOSIS")

    # Phrase matching
    for match_id, start, end in sym_matcher(doc):
        symptoms.add(doc[start:end].text)
    for match_id, start, end in diag_matcher(doc):
        diagnoses.add(doc[start:end].text)
    for match_id, start, end in treat_matcher(doc):
        treatments.add(doc[start:end].text)
    for match_id, start, end in prog_matcher(doc):
        prognosis.add(doc[start:end].text)

    # Numeric session extraction
    session_matches = re.findall(r'(\b\d+\b)\s+(physiotherapy|sessions|session)', text, flags=re.I)
    for num, _ in session_matches:
        treatments.add(f"{num} physiotherapy sessions")
    
    word_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    for w, n in word_numbers.items():
        if re.search(rf'(\b{w}\b)\s+(physiotherapy|sessions|session)', text, flags=re.I):
            treatments.add(f"{n} physiotherapy sessions")

    # Pain mentions
    pain_matches = re.findall(
        r'(\b(neck|back|head|lower back|shoulder|arm|leg)\b).{0,20}\b(pain|ache|aching)\b',
        text, flags=re.I
    )
    for m in pain_matches:
        symptoms.add(f"{m[0]} pain")

    # Head impact detection
    if re.search(r'\bhit my head\b', text, flags=re.I) or \
       re.search(r'\bhead on the steering wheel\b', text, flags=re.I):
        symptoms.add("head impact")

    return {
        "Symptoms": sorted(symptoms),
        "Diagnosis": sorted(diagnoses),
        "Treatment": sorted(treatments),
        "Prognosis": sorted(prognosis),
    }


def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """Extract keywords - uses KeyBERT if available, else fallback"""
    kw_model = get_kw_model()
    
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 3), 
                stop_words='english', top_n=top_n
            )
            return [kw for kw, score in keywords]
        except Exception:
            pass
    
    # Fallback: frequency-based
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    freq = {}
    stopwords = {"the", "and", "you", "patient", "doctor", "i", "a", "to", "was", "that"}
    for w in words:
        if w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:top_n]]


def sentiment_intent_for_utterance(utterance: str) -> Dict[str, Any]:
    """Analyze sentiment and intent"""
    sentiment_pipeline = get_sentiment_pipeline()
    
    # Sentiment analysis
    sentiment = "Neutral"
    score = 0.0
    
    if sentiment_pipeline:
        try:
            res = sentiment_pipeline(utterance[:512])[0]
            label = res['label']
            score = float(res.get('score', 0.0))
            
            if label.upper() == "NEGATIVE":
                sentiment = "Anxious"
            elif label.upper() == "POSITIVE":
                sentiment = "Reassured" if score > 0.9 else "Neutral"
        except Exception:
            pass
    
    # Rule-based intent
    u = utterance.lower()
    if any(w in u for w in ["worried", "worry", "anxious", "concerned", "nervous", "scared", "afraid"]):
        intent = "Seeking reassurance"
    elif any(w in u for w in ["how long", "will i", "affect me", "future", "long-term", "long term"]):
        intent = "Seeking reassurance"
    elif any(w in u for w in ["pain", "hurt", "injury", "accident", "stiffness", "ache", "backache"]):
        intent = "Reporting symptoms"
    elif any(w in u for w in ["thank", "thanks", "appreciate"]):
        intent = "Expressing gratitude"
    else:
        intent = "Other"

    return {"Sentiment": sentiment, "Score": score, "Intent": intent}


def parse_transcript_to_utterances(transcript: str) -> List[Dict[str, str]]:
    """Parse transcript into speaker utterances"""
    utterances = []
    for line in transcript.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(Doctor|Physician|Dr|Patient)\s*:\s*(.*)$', line, flags=re.I)
        if m:
            speaker = m.group(1).capitalize()
            text = m.group(2).strip()
            utterances.append({"speaker": speaker, "text": text})
        else:
            if utterances:
                utterances[-1]["text"] += " " + line
            else:
                utterances.append({"speaker": "Other", "text": line})
    return utterances


def structured_summary_from_transcript(transcript: str) -> Dict[str, Any]:
    """Build structured JSON summary"""
    utterances = parse_transcript_to_utterances(transcript)
    full_text = " ".join([u["text"] for u in utterances])
    entities = extract_entities(full_text)
    keywords = extract_keywords(full_text, top_n=10)

    # Extract patient name
    nlp = get_nlp()
    doc = nlp(full_text)
    patient_name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            patient_name = ent.text
            break

    # Current status
    current_status = None
    for u in reversed(utterances):
        if u["speaker"].lower() == "patient":
            t = u["text"].lower()
            if any(w in t for w in ["now", "currently", "occasion", "sometimes", "still"]):
                current_status = u["text"]
                break

    # Prognosis
    prognosis = entities.get("Prognosis", [])
    if not prognosis:
        m = re.search(r'full recovery.*?within\s+(\w+\s*\w*)', full_text, flags=re.I)
        if m:
            prognosis = [f"Full recovery expected within {m.group(1)}"]

    return {
        "Patient_Name": patient_name or "",
        "Symptoms": entities.get("Symptoms", []),
        "Diagnosis": entities.get("Diagnosis", []),
        "Treatment": entities.get("Treatment", []),
        "Current_Status": current_status or "",
        "Prognosis": prognosis,
        "Keywords": keywords,
    }


def generate_soap_from_transcript(transcript: str) -> Dict[str, Any]:
    """Generate SOAP note"""
    utterances = parse_transcript_to_utterances(transcript)
    full_text = " ".join([u["text"] for u in utterances])
    entities = extract_entities(full_text)

    # Subjective
    chief = ""
    hpi_sentences = []
    for u in utterances:
        if u["speaker"].lower() == "patient":
            text = u["text"]
            if not chief and any(w in text.lower() for w in ["pain", "accident", "hurt", "ache", "whiplash"]):
                chief = text.split(".")[0]
            hpi_sentences.append(text)

    subjective = {
        "Chief_Complaint": chief or ", ".join(entities.get("Symptoms", [])),
        "History_of_Present_Illness": " ".join(hpi_sentences)
    }

    # Objective
    physical_exam = ""
    observations = ""
    for u in utterances:
        if u["speaker"].lower() in ("doctor", "physician", "dr"):
            t = u["text"].lower()
            if "physical examination" in t or "everything looks" in t or "range of motion" in t:
                physical_exam += u["text"] + " "
            if "no tenderness" in t or "no signs" in t or "full range" in t:
                observations += u["text"] + " "

    if not physical_exam:
        m = re.search(r'(full range of movement.*?\.?)', full_text, flags=re.I)
        if m:
            physical_exam = m.group(1)

    if not observations and physical_exam:
        observations = physical_exam

    objective = {
        "Physical_Exam": physical_exam.strip(),
        "Observations": observations.strip()
    }

    # Assessment
    diagnoses = entities.get("Diagnosis", [])
    severity = "Mild"
    if "whiplash" in " ".join([d.lower() for d in diagnoses]):
        severity = "Mild to Moderate"
    if any(w in full_text.lower() for w in ["severe", "hospital", "fracture", "surgery"]):
        severity = "Severe"

    assessment = {
        "Diagnosis": diagnoses,
        "Severity": severity
    }

    # Plan
    plan_treat = []
    follow_up = "Return if symptoms worsen or persist."
    if entities.get("Treatment"):
        plan_treat = entities.get("Treatment")
        plan_treat.append("Use analgesics as needed")
    else:
        plan_treat = ["Conservative management (analgesics, rest)."]

    plan = {
        "Treatment": plan_treat,
        "Follow_Up": follow_up
    }

    return {
        "Subjective": subjective,
        "Objective": objective,
        "Assessment": assessment,
        "Plan": plan
    }


def analyze_transcript(transcript: str) -> Dict[str, Any]:
    """Main analysis function"""
    utterances = parse_transcript_to_utterances(transcript)
    patient_utts = [u for u in utterances if u["speaker"].lower() == "patient"]
    
    sentiment_intent = []
    for u in patient_utts:
        si = sentiment_intent_for_utterance(u["text"])
        sentiment_intent.append({"utterance": u["text"], **si})

    structured = structured_summary_from_transcript(transcript)
    soap = generate_soap_from_transcript(transcript)
    entities = extract_entities(transcript)
    keywords = extract_keywords(transcript, top_n=10)

    return {
        "entities": entities,
        "keywords": keywords,
        "sentiment_intent": sentiment_intent,
        "structured_summary": structured,
        "soap": soap
    }
