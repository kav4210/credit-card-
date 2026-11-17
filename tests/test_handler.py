import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def handler(tmp_path_factory):
    art = tmp_path_factory.mktemp("art")
    here = Path(__file__).parent
    subprocess.run([sys.executable, str(here.parent / "train.py"), "--synthetic", "--out", str(art)], check=True,
                   capture_output=True)
    import os
    os.environ["ARTIFACT_DIR"] = str(art)
    sys.path.insert(0, str(here))
    import importlib
    import handler as h
    importlib.reload(h)
    return h, art


def test_scores_a_transaction(handler):
    h, art = handler
    meta = json.loads((art / "threshold.json").read_text())
    feats = {f: 0.0 for f in meta["features"]}
    res = h.lambda_handler({"features": feats})
    body = json.loads(res["body"])
    assert res["statusCode"] == 200
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["flagged"], bool)


def test_api_gateway_style_event(handler):
    h, art = handler
    meta = json.loads((art / "threshold.json").read_text())
    feats = {f: 0.1 for f in meta["features"]}
    res = h.lambda_handler({"body": json.dumps({"features": feats})})
    assert res["statusCode"] == 200


def test_rejects_missing_features(handler):
    h, _ = handler
    res = h.lambda_handler({"features": {"V1": 1.0}})
    assert res["statusCode"] == 400


def test_rejects_garbage(handler):
    h, _ = handler
    assert h.lambda_handler({"body": "not json"})["statusCode"] == 400
