import pandas as pd

from src.clinical_features import build_radiology_features_from_impression


def test_radiology_feature_extraction_detects_expected_findings():
    df = pd.DataFrame(
        {
            "impression": [
                "Patchy right lower lobe pneumonia",
                "Mild bibasilar atelectasis",
                "Findings suggest small airways disease",
            ]
        }
    )

    cfg = {
        "data": {
            "text_col": "impression",
        },
        "radiology_features": {
            "categories": [
                "pneumonia",
                "atelectasis",
                "small_airways_disease",
            ],
            "synonym_map": {},
        },
        "text_processing": {
            "matching": {
                "use_word_boundaries": True,
            },
            "negation": {
                "enabled": False,
            },
            "radiology_terms": {
                "pneumonia": ["pneumonia"],
                "atelectasis": ["atelectasis"],
                "small_airways_disease": ["small airways disease"],
            },
        },
    }

    out = build_radiology_features_from_impression(df, cfg)

    assert out.loc[0, "pneumonia"] == 1
    assert out.loc[1, "atelectasis"] == 1
    assert out.loc[2, "small_airways_disease"] == 1

def test_radiology_feature_extraction_respects_simple_negation():
    df = pd.DataFrame(
        {
            "impression": [
                "No pneumonia is seen",
                "There is pneumonia in the right lower lobe",
            ]
        }
    )

    cfg = {
        "data": {
            "text_col": "impression",
        },
        "radiology_features": {
            "categories": ["pneumonia"],
            "synonym_map": {},
        },
        "text_processing": {
            "matching": {
                "use_word_boundaries": True,
            },
            "negation": {
                "enabled": True,
                "window": 3,
                "cues": ["no", "without"],
            },
            "radiology_terms": {
                "pneumonia": ["pneumonia"],
            },
        },
    }

    out = build_radiology_features_from_impression(df, cfg)

    assert out.loc[0, "pneumonia"] == 0
    assert out.loc[1, "pneumonia"] == 1