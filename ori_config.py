"""Paths used across the ORI pipeline.

Every path can be overridden with an environment variable, so the scripts
work on any machine without editing code. Defaults point inside the repo.

    ORI_OUTPUTS_DIR      processed data per match (tracking, events, pitch control)
    ORI_RAW_DATA_DIR     raw season folders, e.g. "ESP I 2024/<match_id>"
    ORI_MATCH_LISTS_DIR  CSV lists of matches to process
    ORI_FIGURES_DIR      where plots are saved
    ORI_S3_FETCH_SCRIPT  helper that downloads one match file from storage
    ORI_UPLOAD_SCRIPT    helper that uploads processed match files to storage

The storage helpers are not part of this repository: the tracking and event
data used in the thesis is proprietary and was accessed through the data
provider's own storage.
"""

import os
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent)

OUTPUTS_DIR = os.environ.get("ORI_OUTPUTS_DIR", os.path.join(REPO_ROOT, "outputs"))
RAW_DATA_DIR = os.environ.get("ORI_RAW_DATA_DIR", os.path.join(REPO_ROOT, "data", "raw"))
MATCH_LISTS_DIR = os.environ.get("ORI_MATCH_LISTS_DIR", os.path.join(REPO_ROOT, "data"))
FIGURES_DIR = os.environ.get("ORI_FIGURES_DIR", os.path.join(REPO_ROOT, "docs", "figures"))

S3_FETCH_SCRIPT = os.environ.get("ORI_S3_FETCH_SCRIPT", os.path.join(REPO_ROOT, "scripts", "s3_get_file.py"))
UPLOAD_SCRIPT = os.environ.get("ORI_UPLOAD_SCRIPT", os.path.join(REPO_ROOT, "scripts", "upload_matches.py"))
