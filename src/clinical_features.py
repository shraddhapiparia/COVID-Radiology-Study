from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import deep_get

# ----------------------------
# Utilities
# ----------------------------

def _norm_text(s: Any) -> str:
    return " ".join(str(s).lower().split()) if s is not None else ""


def _contains_phrase(text: str, phrase: str, *, word_boundaries: bool = True) -> bool:
    """
    Conservative phrase match. If word_boundaries=True, uses simple boundary checks to avoid
    matching 'line' inside 'baseline'. Not regex-heavy to keep dependencies low.
    """
    t = _norm_text(text)
    p = _norm_text(phrase)
    if not p:
        return False
    if not word_boundaries:
        return p in t

    # boundary-ish checks
    # add spaces to reduce partial matches
    t2 = f" {t} "
    p2 = f" {p} "
    return p2 in t2


def _apply_synonym_map(terms: Iterable[str], synonym_map: Dict[str, str]) -> List[str]:
    out = []
    for term in terms:
        k = _norm_text(term)
        out.append(synonym_map.get(k, term))
    return out


# ----------------------------
# Clinical NER + Negation (optional)
# ----------------------------

@dataclass
class NERBackend:
    nlp: Any
    use_negation: bool


def _try_load_ner_backend(cfg: Dict[str, Any]) -> Optional[NERBackend]:
    """
    Optional: tries to create a clinical NER pipeline.
    If not installed, returns None and we fall back to phrase matching.
    """
    ner_cfg = deep_get(cfg, "text_processing.ner", {})
    if not isinstance(ner_cfg, dict) or not ner_cfg.get("enabled", False):
        return None

    engine = ner_cfg.get("engine", "spacy_stanza")
    use_neg = bool(deep_get(cfg, "text_processing.ner.negation.enabled", False))

    try:
        if engine == "spacy_stanza":
            import spacy
            import spacy_stanza  # noqa: F401

            # Many environments need explicit stanza download; we won't do downloads here.
            # User should ensure their notebook/env already has models.
            nlp = spacy.blank("en")
            # If the user has a fully configured spacy-stanza pipeline, they can wire it here.
            # We keep this minimal and rely on fallback matching unless user customizes.
            return NERBackend(nlp=nlp, use_negation=use_neg)

        # Could add other engines here
        return None
    except Exception:
        return None


def _negated_by_window(text: str, start: int, *, window: int, cues: List[str]) -> bool:
    """
    Lightweight negation heuristic: check for negation cues within N tokens before the entity span.
    This is a fallback; you can replace with negspacy if installed.
    """
    toks = _norm_text(text).split()
    if not toks:
        return False
    i0 = max(0, start - window)
    ctx = toks[i0:start]
    cue_set = set(_norm_text(c) for c in cues)
    return any(tok in cue_set for tok in ctx)


# ----------------------------
# Radiology features from impression text
# ----------------------------

def build_radiology_features_from_impression(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Produces binary columns for radiology categories.
    Uses:
      - If configured and backend available: basic entity extraction (placeholder) + negation window
      - Else: phrase matching directly from YAML lexicon / category list

    Expected YAML:
      radiology_features.categories (or desired_feature_order)
      radiology_features.synonym_map
      text_processing.negation.enabled/window/cues (for fallback negation)
      text_processing.matching.use_word_boundaries
      text_processing.radiology_terms OR radiology_features.categories

    """
    text_col = deep_get(cfg, "data.text_col")
    if not text_col or text_col not in df.columns:
        raise ValueError(f"Missing text column '{text_col}' for radiology feature extraction.")

    categories = (
        deep_get(cfg, "radiology_features.desired_feature_order")
        or deep_get(cfg, "radiology_features.categories")
        or []
    )
    if not categories:
        # allow older config where radiology_terms dict exists
        rad_terms = deep_get(cfg, "text_processing.radiology_terms", {})
        if isinstance(rad_terms, dict) and rad_terms:
            categories = list(rad_terms.keys())
        else:
            raise ValueError("No radiology categories configured (radiology_features.* or text_processing.radiology_terms).")

    synonym_map = deep_get(cfg, "radiology_features.synonym_map", {}) or {}
    synonym_map = { _norm_text(k): v for k, v in synonym_map.items() }

    # Phrase lexicon:
    # If radiology_terms is defined as dict {category: {phrases:[...]}} or {category:[...]}
    rad_terms = deep_get(cfg, "text_processing.radiology_terms", {})
    lexicon: Dict[str, List[str]] = {}
    if isinstance(rad_terms, dict) and rad_terms:
        for k, v in rad_terms.items():
            if isinstance(v, dict) and "phrases" in v:
                lexicon[k] = list(v.get("phrases") or [])
            elif isinstance(v, list):
                lexicon[k] = list(v)
    # If no explicit lexicon, use category name itself
    for c in categories:
        lexicon.setdefault(c, [c])

    # Matching + negation config
    use_word_boundaries = bool(deep_get(cfg, "text_processing.matching.use_word_boundaries", True))
    neg_enabled = bool(deep_get(cfg, "text_processing.negation.enabled", False))
    neg_window = int(deep_get(cfg, "text_processing.negation.window", 6))
    neg_cues = deep_get(cfg, "text_processing.negation.cues", []) or []

    # Optional backend (not required)
    backend = _try_load_ner_backend(cfg)

    out = df.copy()
    for c in categories:
        out[c] = 0

    texts = out[text_col].fillna("").astype(str).tolist()

    # Fallback: phrase matching per category, with simple negation window
    for idx, raw in enumerate(texts):
        t = _norm_text(raw)
        if not t:
            continue

        for cat in categories:
            phrases = lexicon.get(cat, [cat])
            hit = False
            for ph in phrases:
                if _contains_phrase(t, ph, word_boundaries=use_word_boundaries):
                    if neg_enabled:
                        # crude: if phrase appears, estimate "start token index" for negation window
                        # using first occurrence
                        toks = t.split()
                        ph_toks = _norm_text(ph).split()
                        start = None
                        for i in range(0, max(1, len(toks) - len(ph_toks) + 1)):
                            if toks[i:i+len(ph_toks)] == ph_toks:
                                start = i
                                break
                        if start is not None and _negated_by_window(t, start, window=neg_window, cues=neg_cues):
                            continue
                    hit = True
                    break

            if hit:
                out.iat[idx, out.columns.get_loc(cat)] = 1

    # Apply synonym normalization (not for column names, but if you later export terms)
    # Here we keep columns as canonical category names. If synonym_map collapses multiple categories,
    # you can optionally merge them:
    merged = {}
    for c in categories:
        canonical = synonym_map.get(_norm_text(c), c)
        merged.setdefault(canonical, []).append(c)

    if any(len(v) > 1 for v in merged.values()):
        # collapse columns that map to same canonical
        for canon, cols in merged.items():
            if len(cols) == 1:
                continue
            out[canon] = out[cols].max(axis=1)
            for col in cols:
                if col != canon:
                    out.drop(columns=[col], inplace=True)

    return out


# ----------------------------
# Symptoms features from ROS table
# ----------------------------

def build_symptom_features_from_ros(
    ros_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    id_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Converts a long ROS table into a wide feature table.

    Expected ROS schema (typical):
      - id_col (pat_id)
      - a symptom name column (e.g., 'Symptom' / 'ros_item')
      - a value column (0/1 or True/False)

    YAML expectations:
      symptoms.rename_map: {raw_name -> short_name}
      symptoms.feature_cols: list of final columns
    """
    id_col = id_col_override or deep_get(cfg, "data.id_col", "pat_id")
    rename_map = deep_get(cfg, "symptoms.rename_map", {}) or {}
    target_cols = deep_get(cfg, "symptoms.feature_cols", []) or []

    # Heuristic detection of name/value columns
    # Prefer explicit columns if user added them to YAML later
    name_col = deep_get(cfg, "symptoms.name_col", None)
    val_col = deep_get(cfg, "symptoms.value_col", None)

    if name_col is None:
        # guess
        for cand in ["symptom", "Symptom", "ROS_ITEM", "ros_item", "item", "name"]:
            if cand in ros_df.columns:
                name_col = cand
                break
    if val_col is None:
        for cand in ["value", "Value", "present", "is_present", "flag", "y"]:
            if cand in ros_df.columns:
                val_col = cand
                break

    if name_col is None or val_col is None:
        raise ValueError("Could not infer ROS name/value columns. Set symptoms.name_col and symptoms.value_col in YAML.")

    tmp = ros_df[[id_col, name_col, val_col]].copy()
    tmp[name_col] = tmp[name_col].map(lambda x: rename_map.get(x, x))
    # normalize to 0/1
    tmp[val_col] = tmp[val_col].apply(lambda v: 1 if (v is True or str(v).strip() in {"1", "True", "true", "YES", "yes"}) else 0)

    wide = tmp.pivot_table(index=id_col, columns=name_col, values=val_col, aggfunc="max", fill_value=0)
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()

    # Ensure all expected symptom columns exist
    for c in target_cols:
        if c not in wide.columns:
            wide[c] = 0

    keep = [id_col] + [c for c in target_cols if c in wide.columns]
    return wide[keep]


# ----------------------------
# Demographics derived features
# ----------------------------

def build_demographics_features(
    demo_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    id_col_override: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds derived demographic flags commonly used in your notebook:
      IsHispanic, IsMale, etc + age threshold feature (age-lt-mean).
    """
    id_col = id_col_override or deep_get(cfg, "data.id_col", "pat_id")
    age_col = deep_get(cfg, "demographics.age_col", "age")
    sex_col = deep_get(cfg, "demographics.sex_col", "sex")
    eth_col = deep_get(cfg, "demographics.ethnicity_col", "ethnicity")
    race_col = deep_get(cfg, "demographics.race_col", "race")

    age_threshold = float(deep_get(cfg, "demographics.age_threshold", 8.12))

    df = demo_df.copy()
    if id_col not in df.columns:
        raise ValueError(f"Demographics table missing id_col='{id_col}'")

    out = df[[id_col]].copy()

    # Age
    if age_col in df.columns:
        age = pd.to_numeric(df[age_col], errors="coerce")
        out["age-lt-mean"] = (age <= age_threshold).astype(int)
        out["age_years"] = age
    else:
        out["age-lt-mean"] = 0

    # Sex
    if sex_col in df.columns:
        s = df[sex_col].astype(str).str.lower()
        out["IsMale"] = s.isin(["m", "male"]).astype(int)
        out["IsFemale"] = s.isin(["f", "female"]).astype(int)
    else:
        out["IsMale"] = 0
        out["IsFemale"] = 0

    # Ethnicity (Hispanic)
    if eth_col in df.columns:
        e = df[eth_col].astype(str).str.lower()
        out["IsHispanic"] = e.str.contains("hisp", na=False).astype(int)
        out["IsNonHispanic"] = (~e.str.contains("hisp", na=False)).astype(int)
    else:
        out["IsHispanic"] = 0

    # Race one-hots (optional)
    if race_col in df.columns:
        r = df[race_col].astype(str).str.lower()
        out["IsWhite"] = r.str.contains("white", na=False).astype(int)
        out["IsBlack"] = r.str.contains("black|african", na=False).astype(int)
        out["IsAsian"] = r.str.contains("asian", na=False).astype(int)
        out["IsNHPI"] = r.str.contains("hawaiian|pacific", na=False).astype(int)
        out["IsOther"] = (~(out["IsWhite"].astype(bool) | out["IsBlack"].astype(bool) | out["IsAsian"].astype(bool) | out["IsNHPI"].astype(bool))).astype(int)

    return out


# ----------------------------
# History features (problem list buckets)
# ----------------------------

def build_history_features(
    hist_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    id_col_override: Optional[str] = None,
) -> pd.DataFrame:
    id_col = id_col_override or deep_get(cfg, "data.id_col", "pat_id")
    cols = deep_get(cfg, "history.feature_cols", []) or []
    if id_col not in hist_df.columns:
        raise ValueError(f"History table missing id_col='{id_col}'")

    out = hist_df[[id_col]].copy()
    for c in cols:
        if c in hist_df.columns:
            out[c] = hist_df[c].fillna(0).astype(int)
        else:
            out[c] = 0
    return out


# ----------------------------
# Merge engineered features into one modeling table
# ----------------------------

def assemble_modeling_table(
    base_df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    ros_df: Optional[pd.DataFrame] = None,
    demo_df: Optional[pd.DataFrame] = None,
    hist_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Creates a single table containing:
      - id_col, target_col, optional variant/covid cols
      - radiology feature columns from impression text
      - symptom columns (if provided)
      - demographic derived columns (if provided)
      - history bucket columns (if provided)
    """
    id_col = deep_get(cfg, "data.id_col", "pat_id")
    target_col = deep_get(cfg, "data.target_col")
    covid_col = deep_get(cfg, "data.covid_status_col", None)
    variant_col = deep_get(cfg, "data.variant_col", None)

    df = base_df.copy()
    if id_col not in df.columns:
        raise ValueError(f"Base table missing id_col='{id_col}'")

    # Radiology features from impression text
    df = build_radiology_features_from_impression(df, cfg)

    # Merge symptoms
    if ros_df is not None and deep_get(cfg, "symptoms.enabled", False):
        ros_wide = build_symptom_features_from_ros(ros_df, cfg, id_col_override=id_col)
        df = df.merge(ros_wide, on=id_col, how="left")

    # Merge demographics
    if demo_df is not None and deep_get(cfg, "demographics.enabled", False):
        demo_feat = build_demographics_features(demo_df, cfg, id_col_override=id_col)
        df = df.merge(demo_feat, on=id_col, how="left")

    # Merge history
    if hist_df is not None and deep_get(cfg, "history.enabled", False):
        hist_feat = build_history_features(hist_df, cfg, id_col_override=id_col)
        df = df.merge(hist_feat, on=id_col, how="left")

    # Keep critical cols near front
    front = [c for c in [id_col, target_col, covid_col, variant_col] if c and c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df
