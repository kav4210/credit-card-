"""AWS Lambda handler that scores a transaction with the trained fraud model.

Works with a direct invoke  {"features": {"V1": 0.1, ..., "Amount": 12.3, "Time": 4000}}
and with API Gateway / Function URL events where the JSON is in event["body"].
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd

ART = Path(os.environ.get("ARTIFACT_DIR", "/var/task"))
_model = joblib.load(ART / "model.joblib")
_meta = json.loads((ART / "threshold.json").read_text())
FEATURES = _meta["features"]
THRESHOLD = float(_meta["threshold"])


def _response(code, body):
    return {"statusCode": code, "headers": {"Content-Type": "application/json"}, "body": json.dumps(body)}


def lambda_handler(event, context=None):
    try:
        payload = json.loads(event["body"]) if isinstance(event.get("body"), str) else event
        feats = payload["features"]
        missing = [f for f in FEATURES if f not in feats]
        if missing:
            return _response(400, {"error": "missing features", "missing": missing[:10]})
        row = pd.DataFrame([[float(feats[f]) for f in FEATURES]], columns=FEATURES)
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return _response(400, {"error": f"bad request ({type(exc).__name__})"})
    proba = float(_model.predict_proba(row)[0, 1])
    return _response(200, {"fraud_probability": round(proba, 6), "flagged": proba >= THRESHOLD,
                           "threshold": round(THRESHOLD, 6), "model": _meta["model"]})
