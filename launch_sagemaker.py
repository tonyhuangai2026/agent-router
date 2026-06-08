#!/usr/bin/env python
"""SageMaker launcher for the Qwen3-1.7B LoRA SFT classifier (Tech Design §4.2).

This builds a HuggingFace ``Estimator`` from ``config.yaml`` (the single source of
truth, Tech Design §4) wiring ``entry_point="train.py" + source_dir="src"`` so the
training job runs ``src/train.py`` and pip-installs ``src/requirements.txt`` on the
HF Deep Learning Container. It uploads the prepared ``train/val/test.jsonl`` to S3
and submits a real TrainingJob — *when the environment allows it*.

Credential / role / bucket auto-detection (Tech Design §4.2)
------------------------------------------------------------
A real ``estimator.fit()`` is attempted ONLY when ALL THREE resolve:

1. **Credentials** — ``aws sts get-caller-identity`` succeeds (callable identity).
2. **Execution role** — resolved from, in priority order:
     a. env ``SAGEMAKER_ROLE``,
     b. ``aws.execution_role`` in config.yaml,
     c. auto-discovery of an ``AmazonSageMaker-ExecutionRole-*`` via IAM list-roles.
3. **Bucket** — an explicit ``s3.bucket`` or the SageMaker default bucket
   (``sagemaker-<region>-<account>``) is reachable, or creatable.

If any leg is missing — OR the real submission raises (e.g. the caller lacks
``sagemaker:CreateTrainingJob`` / ``iam:PassRole``, or the ``sagemaker`` SDK is not
installed) — the launcher prints a precise, copy-pasteable **RUNBOOK** (exact role
ARN, bucket, region, and the env/CLI to set) and **exits 0**. It never crashes the
pipeline. See :func:`main`.

The ``sagemaker`` SDK is imported **lazily** (only on a real launch) so that
``--dry-run`` validates the plan and prints it WITHOUT requiring the SDK or any AWS
credentials. ``boto3`` is likewise optional and imported lazily for detection.

Flags
-----
* ``--dry-run`` — never submit. Validate config and print the FULL job plan
  (image, instance, hyperparameters, S3 paths, role/bucket resolution). Exits 0.
* ``--wait`` — on a real launch, stream the training-job logs (``fit(wait=True)``).
* ``--config PATH`` — alternate config file (default: ``config.yaml`` next to this).

Examples
--------
    # Offline validation — works with no SDK and no AWS creds:
    python launch_sagemaker.py --dry-run

    # Attempt a real launch (auto-detect creds/role/bucket); falls back to a
    # runbook + exit 0 if anything is missing or submission is denied:
    python launch_sagemaker.py

    # Real launch and stream logs until the job finishes:
    python launch_sagemaker.py --wait
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_THIS_DIR, "config.yaml")


# =============================================================================
# Config loading + validation
# =============================================================================

def load_config(path: str) -> Dict[str, Any]:
    """Load and minimally validate ``config.yaml``.

    Raises a clear ``SystemExit`` (not a traceback) on a missing file or missing
    required sections so a misconfiguration is actionable rather than a stack
    dump. PyYAML is a launcher dependency (it is in the launcher requirements
    note); if it is somehow absent we explain how to install it.
    """
    if not os.path.isfile(path):
        sys.exit(f"[launch] ERROR: config file not found: {path}\n"
                 f"         Pass --config <path> or create config.yaml next to this script.")
    try:
        import yaml  # PyYAML
    except ImportError:
        sys.exit("[launch] ERROR: PyYAML is required to read config.yaml.\n"
                 "         Install it with:  pip install pyyaml")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Validate required sections / keys up front (fail fast, with the missing key
    # named) so --dry-run is a genuine config validator, not a smoke screen.
    required = {
        "aws": ["region", "account"],
        "s3": ["prefix", "data_subpath", "output_subpath"],
        "data": ["prepared_dir", "train_file", "validation_file"],
        "compute": ["instance_type", "instance_count"],
        "code": ["source_dir", "entry_point"],
        "image": ["transformers_version", "pytorch_version", "py_version"],
        "hyperparameters": ["model_id", "max_len"],
    }
    problems: List[str] = []
    for section, keys in required.items():
        if section not in cfg or not isinstance(cfg.get(section), dict):
            problems.append(f"missing section: '{section}'")
            continue
        for key in keys:
            if key not in cfg[section] or cfg[section][key] in (None, ""):
                # account/region/instance etc. must be present and non-empty.
                problems.append(f"missing/empty key: '{section}.{key}'")
    if problems:
        sys.exit("[launch] ERROR: config.yaml is invalid:\n  - " + "\n  - ".join(problems))

    return cfg


def _abs_under_config(cfg_dir: str, rel_or_abs: str) -> str:
    """Resolve a path from config relative to the config file's directory."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.normpath(os.path.join(cfg_dir, rel_or_abs))


# =============================================================================
# CloudWatch metric_definitions (loss-curve monitoring)
# =============================================================================

def _default_metric_definitions() -> List[Dict[str, str]]:
    """Built-in metric_definitions used when config omits ``metrics``.

    SageMaker scrapes the training-job log against each ``Regex`` and publishes
    the captured numeric group as a CloudWatch metric (namespace
    ``/aws/sagemaker/TrainingJobs``), giving a live loss curve with no
    wandb/tensorboard. ``src/train.py`` runs the HF Trainer (``logging_steps=5``,
    ``report_to=[]``) whose ``PrinterCallback`` prints each log as a Python dict
    to stdout. The text varies by transformers version:

    * transformers 4.51 (the pinned DLC for the real job) prints the RAW dict with
      UNQUOTED floats:   ``{'loss': 0.1234, 'learning_rate': 0.0002, 'epoch': 0.93}``
    * transformers 5.x formats floats as ``:.4g`` STRINGS (quoted):
      ``{'loss': '0.1234', 'learning_rate': '0.0002', 'epoch': '0.93'}``

    The optional quote ``'?`` after each colon matches BOTH styles; the leading
    ``'`` before the key anchors ``'loss'`` so it does not also fire on
    ``'eval_loss'`` / ``'train_loss'``. Verified against real Trainer output.
    """
    return [
        {"Name": "train:loss", "Regex": r"'loss':\s*'?([0-9\.]+)"},
        {"Name": "eval:loss", "Regex": r"'eval_loss':\s*'?([0-9\.]+)"},
        {"Name": "learning_rate", "Regex": r"'learning_rate':\s*'?([0-9\.eE+-]+)"},
        {"Name": "epoch", "Regex": r"'epoch':\s*'?([0-9\.]+)"},
    ]


def _load_metric_definitions(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """Resolve metric_definitions from ``cfg['metrics']['metric_definitions']``.

    Falls back to :func:`_default_metric_definitions` when the block is absent or
    empty. Each entry must be a ``{"Name": ..., "Regex": ...}`` mapping (the exact
    shape the SageMaker ``HuggingFace`` estimator expects); malformed entries are
    skipped with a warning rather than crashing the launch.
    """
    metrics_cfg = cfg.get("metrics") or {}
    raw = metrics_cfg.get("metric_definitions") if isinstance(metrics_cfg, dict) else None
    if not raw:
        return _default_metric_definitions()
    out: List[Dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, dict) and entry.get("Name") and entry.get("Regex"):
            out.append({"Name": str(entry["Name"]), "Regex": str(entry["Regex"])})
        else:
            print(f"[launch] WARNING: ignoring malformed metric definition: {entry!r}")
    return out or _default_metric_definitions()


# =============================================================================
# Derived plan (pure: no AWS, no SDK) — drives both --dry-run and a real launch
# =============================================================================

class JobPlan:
    """Everything needed to describe (and later submit) the training job.

    Built purely from config + local filesystem; contains NO AWS calls so it is
    safe to construct and print under ``--dry-run`` with no creds and no SDK.
    """

    def __init__(self, cfg: Dict[str, Any], cfg_path: str):
        self.cfg = cfg
        self.cfg_path = cfg_path
        cfg_dir = os.path.dirname(os.path.abspath(cfg_path))

        aws = cfg["aws"]
        s3 = cfg["s3"]
        data = cfg["data"]
        compute = cfg["compute"]
        code = cfg["code"]
        image = cfg["image"]

        self.region: str = str(aws["region"])
        self.account: str = str(aws["account"])
        self.config_role: str = str(aws.get("execution_role", "") or "")
        self.role_name_contains: str = str(
            aws.get("execution_role_name_contains", "AmazonSageMaker-ExecutionRole"))

        # S3: explicit bucket or the SageMaker default for this account/region.
        self.explicit_bucket: str = str(s3.get("bucket", "") or "")
        self.default_bucket: str = f"sagemaker-{self.region}-{self.account}"
        self.bucket: str = self.explicit_bucket or self.default_bucket
        self.prefix: str = str(s3["prefix"]).strip("/")
        self.data_subpath: str = str(s3["data_subpath"]).strip("/")
        self.output_subpath: str = str(s3["output_subpath"]).strip("/")

        # Local prepared data + channel files.
        self.prepared_dir: str = _abs_under_config(cfg_dir, str(data["prepared_dir"]))
        self.train_file: str = str(data["train_file"])
        self.validation_file: str = str(data["validation_file"])
        self.test_file: str = str(data.get("test_file", "") or "")

        # Compute.
        self.instance_type: str = str(compute["instance_type"])
        self.instance_count: int = int(compute["instance_count"])
        self.volume_size_gb: int = int(compute.get("volume_size_gb", 100))
        self.max_run_seconds: int = int(compute.get("max_run_seconds", 86400))
        self.use_spot: bool = bool(compute.get("use_spot_instances", False))
        self.max_wait_seconds: int = int(compute.get("max_wait_seconds", 90000))

        # Code packaging (entry_point + source_dir — the AC-4 wiring).
        self.source_dir: str = _abs_under_config(cfg_dir, str(code["source_dir"]))
        self.entry_point: str = str(code["entry_point"])
        self.base_job_name: str = str(code.get("base_job_name", "qwen-classifier-lora-sft"))

        # Image selection.
        self.transformers_version: str = str(image["transformers_version"])
        self.pytorch_version: str = str(image["pytorch_version"])
        self.py_version: str = str(image["py_version"])
        self.image_uri: str = str(image.get("image_uri", "") or "")

        # Hyperparameters (forwarded verbatim to train.py as --key value).
        self.hyperparameters: Dict[str, Any] = dict(cfg["hyperparameters"])

        # CloudWatch loss-curve monitoring: regexes SageMaker scrapes from the
        # training log to publish metrics into /aws/sagemaker/TrainingJobs. Read
        # from config (metrics.metric_definitions) with a built-in default so the
        # launcher still wires monitoring even if the block is omitted. See
        # _default_metric_definitions for the format rationale (HF Trainer dict).
        self.metric_definitions: List[Dict[str, str]] = _load_metric_definitions(cfg)

    # --- derived S3 URIs -----------------------------------------------------
    @property
    def s3_data_base(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{self.data_subpath}"

    @property
    def s3_train(self) -> str:
        return f"{self.s3_data_base}/train"

    @property
    def s3_validation(self) -> str:
        return f"{self.s3_data_base}/validation"

    @property
    def s3_test(self) -> str:
        return f"{self.s3_data_base}/test"

    @property
    def s3_output(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{self.output_subpath}"

    # --- local-data validation ----------------------------------------------
    def local_channel_files(self) -> Dict[str, str]:
        """Map channel name -> local file path (only channels with a file set)."""
        files = {
            "train": os.path.join(self.prepared_dir, self.train_file),
            "validation": os.path.join(self.prepared_dir, self.validation_file),
        }
        if self.test_file:
            files["test"] = os.path.join(self.prepared_dir, self.test_file)
        return files

    def validate_local_data(self) -> List[str]:
        """Return a list of human-readable problems with the local prepared data."""
        problems: List[str] = []
        if not os.path.isdir(self.prepared_dir):
            problems.append(f"prepared_dir does not exist: {self.prepared_dir}")
            return problems
        for channel, path in self.local_channel_files().items():
            if not os.path.isfile(path):
                problems.append(f"missing {channel} file: {path}")
        return problems

    def validate_code(self) -> List[str]:
        """Return problems with the training code packaging (entry_point/source_dir)."""
        problems: List[str] = []
        if not os.path.isdir(self.source_dir):
            problems.append(f"source_dir does not exist: {self.source_dir}")
            return problems
        entry = os.path.join(self.source_dir, self.entry_point)
        if not os.path.isfile(entry):
            problems.append(f"entry_point not found in source_dir: {entry}")
        reqs = os.path.join(self.source_dir, "requirements.txt")
        if not os.path.isfile(reqs):
            problems.append(
                f"requirements.txt not found in source_dir ({reqs}); the job would "
                f"run without installing peft/trl/etc.")
        return problems


# =============================================================================
# AWS detection (lazy boto3; degrades to AWS CLI; never raises)
# =============================================================================

class Detection:
    """Result of credential / role / bucket auto-detection."""

    def __init__(self) -> None:
        self.identity_ok: bool = False
        self.identity: Optional[Dict[str, str]] = None
        self.identity_error: Optional[str] = None
        self.role_arn: Optional[str] = None
        self.role_source: Optional[str] = None      # how the role was resolved
        self.role_error: Optional[str] = None
        self.bucket: Optional[str] = None
        self.bucket_exists: bool = False
        self.bucket_creatable: bool = False
        self.bucket_error: Optional[str] = None
        self.boto3_available: bool = False

    @property
    def ready(self) -> bool:
        """True only when creds + role + (existing or creatable) bucket all resolve."""
        return bool(
            self.identity_ok
            and self.role_arn
            and self.bucket
            and (self.bucket_exists or self.bucket_creatable)
        )


def _run_cli_json(args: List[str], region: Optional[str] = None,
                  timeout: int = 40) -> Tuple[bool, Any, str]:
    """Run an `aws ...` CLI command returning JSON. Returns (ok, parsed, stderr).

    Used as a credential/identity probe and as a fallback when boto3 is absent.
    Never raises — a missing CLI or a non-zero exit becomes ``(False, None, msg)``.
    """
    if shutil.which("aws") is None:
        return False, None, "aws CLI not found on PATH"
    cmd = ["aws"] + args + ["--output", "json"]
    if region:
        cmd += ["--region", region]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # subprocess failure, timeout, etc.
        return False, None, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return False, None, (proc.stderr or proc.stdout or "non-zero exit").strip()
    out = (proc.stdout or "").strip()
    if not out:
        return True, None, ""
    try:
        return True, json.loads(out), ""
    except json.JSONDecodeError:
        return True, out, ""


def detect_aws(plan: JobPlan, verbose: bool = True) -> Detection:
    """Auto-detect credentials, an execution role, and a usable bucket.

    Order of operations mirrors Tech Design §4.2. Uses boto3 when available and
    falls back to the AWS CLI for the identity probe. Every step is wrapped so a
    permission/credential failure is *recorded*, never raised — the caller decides
    (submit vs. runbook) from :pyattr:`Detection.ready`.
    """
    det = Detection()

    # boto3 is optional; we lazily import and degrade to the CLI for identity.
    boto3 = None
    try:
        import boto3 as _boto3  # noqa: N813
        boto3 = _boto3
        det.boto3_available = True
    except ImportError:
        det.boto3_available = False

    # --- 1. Credentials: sts get-caller-identity ----------------------------
    if boto3 is not None:
        try:
            sts = boto3.client("sts", region_name=plan.region)
            ident = sts.get_caller_identity()
            det.identity_ok = True
            det.identity = {
                "Account": ident.get("Account", ""),
                "Arn": ident.get("Arn", ""),
                "UserId": ident.get("UserId", ""),
            }
        except Exception as exc:
            det.identity_error = f"{type(exc).__name__}: {exc}"
    if not det.identity_ok:
        # boto3 missing or sts call failed → try the CLI as a second source.
        ok, parsed, err = _run_cli_json(["sts", "get-caller-identity"], plan.region)
        if ok and isinstance(parsed, dict):
            det.identity_ok = True
            det.identity = {
                "Account": parsed.get("Account", ""),
                "Arn": parsed.get("Arn", ""),
                "UserId": parsed.get("UserId", ""),
            }
            det.identity_error = None
        elif det.identity_error is None:
            det.identity_error = err or "could not call sts get-caller-identity"

    if verbose:
        if det.identity_ok and det.identity:
            print(f"[detect] credentials OK — caller {det.identity.get('Arn')}")
        else:
            print(f"[detect] credentials NOT available — {det.identity_error}")

    # --- 2. Execution role ---------------------------------------------------
    det.role_arn, det.role_source, det.role_error = _resolve_role(plan, boto3, verbose)

    # --- 3. Bucket -----------------------------------------------------------
    _resolve_bucket(plan, boto3, det, verbose)

    return det


def _resolve_role(plan: JobPlan, boto3, verbose: bool
                  ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve the execution-role ARN per the §4.2 priority order.

    Returns (role_arn, source, error). ``source`` is one of
    ``env SAGEMAKER_ROLE`` / ``config aws.execution_role`` / ``IAM auto-discover``.
    """
    # a. env SAGEMAKER_ROLE
    env_role = os.environ.get("SAGEMAKER_ROLE", "").strip()
    if env_role:
        if verbose:
            print(f"[detect] role from env SAGEMAKER_ROLE = {env_role}")
        return env_role, "env SAGEMAKER_ROLE", None

    # b. config aws.execution_role
    if plan.config_role:
        if verbose:
            print(f"[detect] role from config aws.execution_role = {plan.config_role}")
        return plan.config_role, "config aws.execution_role", None

    # c. auto-discover an AmazonSageMaker-ExecutionRole-* via IAM
    needle = plan.role_name_contains
    arns: List[str] = []
    err: Optional[str] = None
    if boto3 is not None:
        try:
            iam = boto3.client("iam", region_name=plan.region)
            paginator = iam.get_paginator("list_roles")
            for page in paginator.paginate():
                for role in page.get("Roles", []):
                    if needle in role.get("RoleName", ""):
                        arns.append(role["Arn"])
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    if not arns:
        # CLI fallback (also covers boto3 absent).
        ok, parsed, cli_err = _run_cli_json(
            ["iam", "list-roles",
             "--query", f"Roles[?contains(RoleName, `{needle}`)].Arn"],
            region=plan.region)
        if ok and isinstance(parsed, list):
            arns = [a for a in parsed if isinstance(a, str)]
            err = None
        elif err is None:
            err = cli_err or "could not list IAM roles"

    if arns:
        chosen = sorted(arns)[0]  # deterministic pick
        if verbose:
            print(f"[detect] role auto-discovered via IAM ({len(arns)} match"
                  f"{'es' if len(arns) != 1 else ''}); using {chosen}")
        return chosen, "IAM auto-discover", None

    if verbose:
        print(f"[detect] execution role NOT resolved — {err}")
    return None, None, err or "no SageMaker execution role found"


def _resolve_bucket(plan: JobPlan, boto3, det: Detection, verbose: bool) -> None:
    """Determine the bucket and whether it exists / is creatable. Mutates ``det``."""
    det.bucket = plan.bucket  # explicit or sagemaker-<region>-<account>

    # Need credentials to check the bucket at all.
    if not det.identity_ok:
        det.bucket_error = "skipped (no credentials)"
        if verbose:
            print(f"[detect] bucket check skipped (no credentials): {det.bucket}")
        return

    exists = False
    err: Optional[str] = None
    if boto3 is not None:
        try:
            s3 = boto3.client("s3", region_name=plan.region)
            s3.head_bucket(Bucket=det.bucket)
            exists = True
        except Exception as exc:
            # head_bucket 404 => missing; 403 => exists but not ours / no perm.
            msg = str(exc)
            if "404" in msg or "Not Found" in msg:
                exists = False
            elif "403" in msg or "Forbidden" in msg:
                # The bucket exists; we may still be able to write under our prefix.
                exists = True
            else:
                err = f"{type(exc).__name__}: {exc}"
    else:
        ok, _parsed, cli_err = _run_cli_json(
            ["s3api", "head-bucket", "--bucket", det.bucket], region=plan.region)
        if ok:
            exists = True
        else:
            low = (cli_err or "").lower()
            if "404" in low or "not found" in low:
                exists = False
            elif "403" in low or "forbidden" in low:
                exists = True
            else:
                err = cli_err

    det.bucket_exists = exists
    det.bucket_error = err
    # The SageMaker SDK will create the *default* bucket on demand; treat a
    # missing default bucket as creatable (the SDK / our explicit create handles
    # it). An explicit, missing bucket is also creatable if we have credentials.
    det.bucket_creatable = (not exists) and det.identity_ok

    if verbose:
        if exists:
            print(f"[detect] bucket OK — {det.bucket} (exists)")
        elif det.bucket_creatable:
            print(f"[detect] bucket {det.bucket} missing but creatable "
                  f"(will be created at launch)")
        else:
            print(f"[detect] bucket NOT usable — {det.bucket}: {det.bucket_error}")


# =============================================================================
# Plan printing
# =============================================================================

def _image_description(plan: JobPlan) -> str:
    if plan.image_uri:
        return f"(explicit) {plan.image_uri}"
    return (f"HuggingFace PyTorch training DLC — transformers {plan.transformers_version}, "
            f"pytorch {plan.pytorch_version}, {plan.py_version} "
            f"(resolved by the HuggingFace Estimator for region {plan.region})")


def print_plan(plan: JobPlan, det: Optional[Detection], mode: str) -> None:
    """Print the FULL job plan: image, instance, hyperparameters, S3 paths, role."""
    role_line = "(resolved at launch)"
    bucket_line = plan.bucket + (" [explicit]" if plan.explicit_bucket
                                 else " [SageMaker default]")
    if det is not None:
        role_line = (f"{det.role_arn}  (via {det.role_source})"
                     if det.role_arn else "NOT RESOLVED")
        if det.bucket:
            state = ("exists" if det.bucket_exists
                     else "creatable" if det.bucket_creatable else "unusable")
            bucket_line = f"{det.bucket} [{state}]"

    bar = "=" * 78
    print(bar)
    print(f" SageMaker Training Job Plan  ({mode})")
    print(bar)
    print(f" config file        : {plan.cfg_path}")
    print(f" base job name      : {plan.base_job_name}")
    print()
    print(" -- Compute -------------------------------------------------------")
    print(f" instance_type      : {plan.instance_type}")
    print(f" instance_count     : {plan.instance_count}")
    print(f" volume_size_gb     : {plan.volume_size_gb}")
    print(f" max_run_seconds    : {plan.max_run_seconds}")
    print(f" managed spot       : {plan.use_spot}"
          + (f" (max_wait={plan.max_wait_seconds}s)" if plan.use_spot else ""))
    print()
    print(" -- Image (HF DLC) ------------------------------------------------")
    print(f" image              : {_image_description(plan)}")
    print()
    print(" -- Code packaging ------------------------------------------------")
    print(f" entry_point        : {plan.entry_point}")
    print(f" source_dir         : {plan.source_dir}")
    print(f"   -> the job runs  : {os.path.join('src', plan.entry_point)} "
          f"(== entry_point in source_dir)")
    print(f"   -> requirements  : {os.path.join(plan.source_dir, 'requirements.txt')} "
          f"(pip-installed in the job)")
    print()
    print(" -- AWS placement -------------------------------------------------")
    print(f" region             : {plan.region}")
    print(f" account            : {plan.account}")
    print(f" execution role     : {role_line}")
    print()
    print(" -- S3 paths ------------------------------------------------------")
    print(f" bucket             : {bucket_line}")
    print(f" input  train       : {plan.s3_train}/")
    print(f" input  validation  : {plan.s3_validation}/")
    if plan.test_file:
        print(f" input  test        : {plan.s3_test}/")
    print(f" output (model)     : {plan.s3_output}/   (model.tar.gz lands here)")
    print()
    print(" -- Local data to upload ------------------------------------------")
    for channel, path in plan.local_channel_files().items():
        exists = "OK" if os.path.isfile(path) else "MISSING"
        print(f" {channel:<11}      : {path}  [{exists}]")
    print()
    print(" -- Hyperparameters (forwarded to train.py as --key value) --------")
    for k in sorted(plan.hyperparameters):
        print(f"   --{k} {plan.hyperparameters[k]}")
    print()
    print(" -- CloudWatch metric_definitions (loss-curve monitoring) ---------")
    print(f"   namespace        : /aws/sagemaker/TrainingJobs "
          f"(dimension TrainingJobName=<job>)")
    if plan.metric_definitions:
        for md in plan.metric_definitions:
            print(f"   {md.get('Name'):<16} : {md.get('Regex')}")
    else:
        print("   (none configured)")
    print(bar)


# =============================================================================
# Runbook (printed when auto-detect / submission cannot proceed)
# =============================================================================

def print_runbook(plan: JobPlan, det: Detection, reason: str) -> None:
    """Print a precise, actionable runbook and (caller) exit 0.

    Lists the exact role ARN, bucket, region, and the env/CLI to set so a human
    can complete the launch. Tailored to whichever leg(s) of the auto-detect
    failed (creds / role / bucket / SDK / submission denial).
    """
    bar = "=" * 78
    print()
    print(bar)
    print(" RUNBOOK — real SageMaker launch was NOT performed")
    print(bar)
    print(f" reason: {reason}")
    print()

    # 1) Credentials
    print(" 1) AWS credentials")
    if det.identity_ok and det.identity:
        print(f"      OK — caller identity: {det.identity.get('Arn')}")
    else:
        print(f"      MISSING — `aws sts get-caller-identity` did not succeed.")
        print(f"      Fix: configure credentials for account {plan.account}, e.g.")
        print(f"           export AWS_ACCESS_KEY_ID=...   export AWS_SECRET_ACCESS_KEY=...")
        print(f"           (or:  aws configure   /   attach an instance role)")
    print()

    # 2) Region
    print(" 2) Region")
    print(f"      Use region: {plan.region}")
    print(f"      Fix (if unset):  export AWS_DEFAULT_REGION={plan.region}")
    print()

    # 3) Execution role
    print(" 3) SageMaker execution role")
    if det.role_arn:
        print(f"      Resolved: {det.role_arn}  (via {det.role_source})")
        print(f"      NOTE: the job also requires the *caller* to have")
        print(f"            sagemaker:CreateTrainingJob and iam:PassRole on this role.")
    else:
        suggested = (plan.config_role
                     or f"arn:aws:iam::{plan.account}:role/service-role/"
                        f"AmazonSageMaker-ExecutionRole-<timestamp>")
        print(f"      NOT RESOLVED ({det.role_error}).")
        print(f"      Fix: set the role explicitly, e.g.")
        print(f"           export SAGEMAKER_ROLE={suggested}")
        print(f"      Discover one with:")
        print(f"           aws iam list-roles \\")
        print(f"             --query \"Roles[?contains(RoleName,"
              f"'{plan.role_name_contains}')].Arn\" --output text")
    print()

    # 4) Bucket
    print(" 4) S3 bucket")
    if det.bucket and (det.bucket_exists or det.bucket_creatable):
        state = "exists" if det.bucket_exists else "will be created at launch"
        print(f"      OK — {det.bucket} ({state})")
    else:
        print(f"      Target bucket: {plan.bucket}")
        print(f"      Fix: create it (or set s3.bucket in config.yaml to an existing one):")
        print(f"           aws s3 mb s3://{plan.bucket} --region {plan.region}")
    print()

    # 5) SDK
    print(" 5) SageMaker Python SDK")
    sdk_ok = _sagemaker_sdk_available()
    if sdk_ok:
        print(f"      OK — `import sagemaker.huggingface` works.")
    else:
        print(f"      NOT installed (only --dry-run works without it).")
        print(f"      Fix:  pip install 'sagemaker>=2.190,<3'")
    print()

    # 6) Re-run line
    print(" 6) Then re-run the real launch")
    print(f"      cd {os.path.dirname(plan.cfg_path)}")
    print(f"      python launch_sagemaker.py            # auto-detect + submit")
    print(f"      python launch_sagemaker.py --wait     # submit and stream logs")
    print()
    print(" (Validate the plan any time WITHOUT AWS or the SDK:")
    print("    python launch_sagemaker.py --dry-run )")
    print(bar)


def _sagemaker_sdk_available() -> bool:
    """True iff the real ``sagemaker.huggingface.HuggingFace`` estimator imports.

    The environment may carry an empty ``sagemaker`` *namespace* package (no
    submodules); that must NOT be mistaken for an installed SDK, so we probe the
    actual estimator import.
    """
    try:
        from sagemaker.huggingface import HuggingFace  # noqa: F401
        return True
    except Exception:
        return False


# =============================================================================
# Real launch (lazy SDK import; all failures -> runbook, exit 0)
# =============================================================================

def build_estimator_and_fit(plan: JobPlan, det: Detection, wait: bool) -> int:
    """Build the HuggingFace Estimator, upload data, and submit the real job.

    Imports the ``sagemaker`` SDK lazily. ANY failure (SDK missing, bucket create
    denied, ``CreateTrainingJob`` / ``PassRole`` denied, network, etc.) is caught
    and converted into a runbook + ``return 0`` — never a crash.
    """
    # --- Lazy SDK import -----------------------------------------------------
    try:
        import sagemaker
        from sagemaker.huggingface import HuggingFace
        from sagemaker.inputs import TrainingInput
    except Exception as exc:
        print(f"[launch] sagemaker SDK not importable ({type(exc).__name__}: {exc}).")
        print_runbook(plan, det,
                      reason="sagemaker Python SDK is not installed "
                             "(install: pip install 'sagemaker>=2.190,<3')")
        return 0

    try:
        import boto3
        boto_sess = boto3.Session(region_name=plan.region)

        # SageMaker session bound to our (possibly to-be-created) bucket.
        sess = sagemaker.Session(
            boto_session=boto_sess,
            default_bucket=det.bucket,
        )

        # Ensure the bucket exists (SDK creates the default bucket on first use,
        # but we make it explicit so an explicit/missing bucket also works).
        if not det.bucket_exists:
            print(f"[launch] ensuring bucket s3://{det.bucket} exists ...")
            # default_bucket() creates the SageMaker default bucket if needed.
            sess.default_bucket()

        # --- Upload prepared data to per-channel S3 prefixes -----------------
        # One file per channel directory so SM mounts it at SM_CHANNEL_<NAME>.
        s3_prefix = f"{plan.prefix}/{plan.data_subpath}"
        channel_uris: Dict[str, str] = {}
        for channel, local_path in plan.local_channel_files().items():
            key_prefix = f"{s3_prefix}/{channel}"
            uri = sess.upload_data(
                path=local_path,
                bucket=det.bucket,
                key_prefix=key_prefix,
            )
            # upload_data returns the s3 URI of the uploaded object; the channel
            # input is its containing prefix.
            channel_uris[channel] = f"s3://{det.bucket}/{key_prefix}"
            print(f"[launch] uploaded {channel}: {local_path} -> {uri}")

        # --- Build the estimator (AC-4: entry_point + source_dir) ------------
        hp = _stringify_hyperparameters(plan.hyperparameters)
        estimator_kwargs: Dict[str, Any] = dict(
            entry_point=plan.entry_point,        # "train.py"
            source_dir=plan.source_dir,          # ".../src" (requirements.txt installed here)
            role=det.role_arn,
            instance_type=plan.instance_type,
            instance_count=plan.instance_count,
            volume_size=plan.volume_size_gb,
            max_run=plan.max_run_seconds,
            base_job_name=plan.base_job_name,
            hyperparameters=hp,
            output_path=plan.s3_output,
            sagemaker_session=sess,
            disable_profiler=True,
            # CloudWatch loss-curve monitoring: SageMaker scrapes the training log
            # against these regexes and publishes them to CloudWatch namespace
            # /aws/sagemaker/TrainingJobs (visible in the SageMaker console's
            # "Metrics" tab and via get-metric-statistics). See JobPlan.metric_definitions.
            metric_definitions=plan.metric_definitions,
        )
        if plan.image_uri:
            # An explicit image_uri pins the exact ECR image and WINS over the
            # framework version tags. However, the HuggingFace estimator still
            # requires py_version as a positional arg (SDK 2.x), so it must be
            # supplied even here. The config keeps transformers/pytorch/py tags
            # aligned to the pinned image, so passing them alongside image_uri is
            # safe (they only feed tagging/metadata; image_uri still selects the
            # actual container).
            estimator_kwargs.update(
                image_uri=plan.image_uri,
                transformers_version=plan.transformers_version,
                pytorch_version=plan.pytorch_version,
                py_version=plan.py_version,
            )
        else:
            estimator_kwargs.update(
                transformers_version=plan.transformers_version,
                pytorch_version=plan.pytorch_version,
                py_version=plan.py_version,
            )
        if plan.use_spot:
            estimator_kwargs.update(
                use_spot_instances=True,
                max_wait=plan.max_wait_seconds,
            )

        estimator = HuggingFace(**estimator_kwargs)

        inputs = {
            ch: TrainingInput(uri) for ch, uri in channel_uris.items()
        }

        print(f"[launch] submitting training job (wait={wait}) ...")
        estimator.fit(inputs=inputs, wait=wait)

        job_name = None
        try:
            job_name = estimator.latest_training_job.name  # type: ignore[attr-defined]
        except Exception:
            pass
        print()
        print("=" * 78)
        print(" REAL TRAINING JOB SUBMITTED")
        print("=" * 78)
        if job_name:
            print(f" job name   : {job_name}")
        print(f" region     : {plan.region}")
        print(f" output     : {plan.s3_output}/")
        print(f" monitor    : aws sagemaker describe-training-job "
              f"--training-job-name {job_name or '<job-name>'} --region {plan.region}")
        print("=" * 78)
        return 0

    except Exception as exc:
        # Any boto/permission/runtime error during a real submission lands here.
        # This is the graceful path for: caller lacks sagemaker:CreateTrainingJob
        # / iam:PassRole, bucket create denied, image lookup failure, etc.
        print(f"[launch] real submission did not complete "
              f"({type(exc).__name__}: {exc}).")
        print_runbook(
            plan, det,
            reason=f"submission attempt failed: {type(exc).__name__}: {exc}")
        return 0


def _stringify_hyperparameters(hp: Dict[str, Any]) -> Dict[str, Any]:
    """Render hyperparameters for the Estimator.

    SageMaker forwards hyperparameters to the entrypoint as ``--key value``.
    train.py parses bools via a permissive ``_bool_arg``, so booleans are sent as
    lowercase ``"true"``/``"false"`` strings (the SDK would otherwise emit Python
    ``True``/``False``). Everything else is passed through; the SDK stringifies it.
    """
    out: Dict[str, Any] = {}
    for k, v in hp.items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        else:
            out[k] = v
    return out


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build + submit the Qwen3-1.7B LoRA SFT SageMaker job "
                    "(auto-detect creds/role/bucket; graceful runbook fallback).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Path to config.yaml.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config and print the full job plan WITHOUT "
                        "submitting (works with no SDK and no AWS credentials).")
    p.add_argument("--wait", action="store_true",
                   help="On a real launch, stream the training-job logs "
                        "(estimator.fit(wait=True)).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # --- Load + validate config (fails fast with an actionable message) ------
    cfg = load_config(args.config)
    plan = JobPlan(cfg, args.config)

    # --- Validate local code + data (problems reported, not fatal for dry-run)
    code_problems = plan.validate_code()
    data_problems = plan.validate_local_data()

    # =====================================================================
    # DRY RUN: validate + print plan, never touch AWS or the SDK. (AC-1)
    # =====================================================================
    if args.dry_run:
        print_plan(plan, det=None, mode="DRY RUN — no submission")
        ok = True
        if code_problems:
            ok = False
            print("\n[dry-run] CODE PACKAGING PROBLEMS:")
            for pr in code_problems:
                print(f"  - {pr}")
        if data_problems:
            ok = False
            print("\n[dry-run] LOCAL DATA PROBLEMS:")
            for pr in data_problems:
                print(f"  - {pr}")
        sdk = _sagemaker_sdk_available()
        print(f"\n[dry-run] sagemaker SDK importable: {sdk}"
              + ("" if sdk else "  (only needed for a REAL launch; "
                                "install: pip install 'sagemaker>=2.190,<3')"))
        if ok:
            print("\n[dry-run] config + code + local data validated OK. "
                  "No job submitted (dry run).")
        else:
            print("\n[dry-run] validation found problems (listed above). "
                  "No job submitted (dry run).")
        # Dry run always exits 0 — it is a validator, not a gate.
        return 0

    # =====================================================================
    # REAL PATH: auto-detect creds/role/bucket; submit or print runbook. (AC-2)
    # =====================================================================
    print("[launch] auto-detecting AWS credentials, execution role, and bucket ...")
    det = detect_aws(plan, verbose=True)

    # Hard local prerequisites: without the code/data we cannot submit anything.
    if code_problems or data_problems:
        print("\n[launch] local prerequisites missing:")
        for pr in code_problems + data_problems:
            print(f"  - {pr}")
        print_runbook(plan, det,
                      reason="local code/data prerequisites are missing "
                             "(see the list above)")
        return 0

    print()
    print_plan(plan, det=det, mode="REAL LAUNCH — auto-detect")

    if not det.ready:
        # Determine the most relevant reason for the runbook header.
        if not det.identity_ok:
            reason = "AWS credentials not available (sts get-caller-identity failed)"
        elif not det.role_arn:
            reason = "no SageMaker execution role resolved"
        else:
            reason = f"S3 bucket not usable: {det.bucket} ({det.bucket_error})"
        print_runbook(plan, det, reason=reason)
        return 0

    # All three legs resolved → attempt the REAL submission. Any failure inside
    # (incl. a missing SDK or a PassRole/CreateTrainingJob denial) is caught and
    # converted to a runbook + exit 0 by build_estimator_and_fit.
    print("\n[launch] credentials + role + bucket all resolved — "
          "attempting REAL submission.")
    return build_estimator_and_fit(plan, det, wait=args.wait)


if __name__ == "__main__":
    sys.exit(main())
