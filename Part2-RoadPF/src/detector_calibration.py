
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .observation_model import ObservationModelParams, load_params
from .road_graph import EXP_ROOT

CONFIGS = os.path.join(EXP_ROOT, "configs")
PROFILE_DIR = os.path.join(CONFIGS, "detector_profiles")
ACTIVE_FILE = os.path.join(PROFILE_DIR, "active.json")
ENV_VAR = "GRAYSENSE_DETECTOR_PROFILE"




DEFAULT_PROFILE = "dnn_event_v3"




CONSISTENCY_TOL = 1e-9

STATUS_PRELIMINARY = "preliminary"
STATUS_FINAL = "final"


class CalibrationError(RuntimeError):
    pass






@dataclass(frozen=True)
class DetectorProfile:

    name: str
    path: str
    doc: dict = field(repr=False)


    @property
    def detector(self) -> str:
        return str(self.doc.get("detector", "unknown"))

    @property
    def detector_version(self) -> str:
        return str(self.doc.get("detector_version", ""))

    @property
    def operating_threshold(self) -> Optional[float]:
        v = self.doc.get("operating_threshold")
        return None if v is None else float(v)

    @property
    def status(self) -> str:
        return str(self.doc.get("status", STATUS_PRELIMINARY))

    @property
    def is_preliminary(self) -> bool:
        return self.status != STATUS_FINAL

    @property
    def description(self) -> str:
        return str(self.doc.get("description", ""))


    @property
    def evaluation(self) -> dict:
        return dict(self.doc.get("evaluation", {}))

    @property
    def event_level(self) -> bool:
        return bool(self.evaluation.get("event_level", False))

    @property
    def precision(self) -> Optional[float]:
        return _opt_float(self.evaluation.get("precision"))

    @property
    def recall(self) -> Optional[float]:
        return _opt_float(self.evaluation.get("recall"))

    @property
    def f1(self) -> Optional[float]:
        return _opt_float(self.evaluation.get("f1"))

    @property
    def source_evaluation(self) -> str:
        return str(self.evaluation.get("source_file", ""))

    @property
    def run_id(self) -> str:
        return str(self.evaluation.get("run_id", ""))


    @property
    def simulation(self) -> dict:
        return dict(self.doc.get("simulation", {}))

    @property
    def false_negative_rate(self) -> Optional[float]:
        return _opt_float(self.simulation.get("false_negative_rate"))

    @property
    def false_positive_rate(self) -> Optional[float]:
        return _opt_float(self.simulation.get("false_positive_rate"))

    @property
    def timing_jitter_std_s(self) -> Optional[float]:
        return _opt_float(self.simulation.get("timing_jitter_std_s"))

    @property
    def timing_bias_s(self) -> Optional[float]:
        return _opt_float(self.simulation.get("timing_bias_s"))

    @property
    def apply_timing_bias(self) -> bool:
        return bool(self.simulation.get("apply_timing_bias", False))

    @property
    def timing_jitter_provenance(self) -> str:
        return str(self.simulation.get("timing_jitter_provenance", "unspecified"))

    @property
    def observation_model_path(self) -> str:
        rel = self.simulation.get("observation_model_config")
        if not rel:
            raise CalibrationError(
                f"profile {self.name!r} names no observation_model_config")
        return rel if os.path.isabs(rel) else os.path.join(EXP_ROOT, rel)


    def params(self, require_event_level: bool = True) -> ObservationModelParams:
        path = self.observation_model_path
        if not os.path.exists(path):
            raise CalibrationError(
                f"profile {self.name!r} points at a missing observation model: {path}")
        params = load_params(path)
        if not params.is_calibrated:
            raise CalibrationError(
                f"profile {self.name!r} -> {path} is not calibrated "
                f"(missing {params.missing_fields()})")

        if require_event_level and not self.event_level:
            raise CalibrationError(
                f"profile {self.name!r} was not evaluated at the event level. "
                "Frame-level precision/recall are not dimensionally compatible "
                "with the simulator's per-passage rates -- re-score the "
                "detector into passages first, or pass "
                "require_event_level=False for a clearly-labelled probe.")

        for attr, got in (("false_negative_rate", params.false_negative_rate),
                          ("false_positive_rate", params.false_positive_rate),
                          ("timing_jitter_std_s", params.timing_jitter_std_s)):
            want = getattr(self, attr)
            if want is None:
                raise CalibrationError(
                    f"profile {self.name!r} declares no simulation.{attr}")
            if abs(float(want) - float(got)) > CONSISTENCY_TOL:
                raise CalibrationError(
                    f"profile {self.name!r} declares {attr}={want!r} but "
                    f"{path} applies {got!r}. Regenerate the profile.")
        return params


    def meta(self) -> dict:
        return {
            "detector_profile": self.name,
            "detector": self.detector,
            "detector_version": self.detector_version,
            "operating_threshold": self.operating_threshold,
            "status": self.status,
            "event_level": self.event_level,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_negative_rate": self.false_negative_rate,
            "false_positive_rate": self.false_positive_rate,
            "timing_jitter_std_s": self.timing_jitter_std_s,
            "timing_bias_s": self.timing_bias_s,
            "apply_timing_bias": self.apply_timing_bias,
            "timing_jitter_provenance": self.timing_jitter_provenance,
            "source_evaluation": self.source_evaluation,
            "run_id": self.run_id,
            "profile_path": os.path.relpath(self.path, EXP_ROOT),
            "observation_model_config": os.path.relpath(
                self.observation_model_path, EXP_ROOT),
        }

    def label(self) -> str:
        thr = ("" if self.operating_threshold is None
               else f" @ {self.operating_threshold:g}")
        star = " (preliminary)" if self.is_preliminary else ""
        return f"{self.detector}{thr}{star}"

    def summary_line(self) -> str:
        def f(x, nd=4):
            return "n/a" if x is None else f"{x:.{nd}f}"
        return (f"[calib] profile={self.name} detector={self.detector} "
                f"thr={self.operating_threshold} status={self.status} "
                f"P={f(self.precision, 3)} R={f(self.recall, 3)} "
                f"F1={f(self.f1, 3)} | p_fn={f(self.false_negative_rate)} "
                f"ghost={f(self.false_positive_rate)} "
                f"sigma={f(self.timing_jitter_std_s)}s "
                f"({self.timing_jitter_provenance})")


def _opt_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, str) and v.strip().upper() == "TODO"):
        return None
    return float(v)






def list_profiles() -> list[str]:
    if not os.path.isdir(PROFILE_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(PROFILE_DIR)
                  if f.endswith(".json") and f != "active.json")


def active_profile_name() -> str:
    env = os.environ.get(ENV_VAR)
    if env:
        return env.strip()
    if os.path.exists(ACTIVE_FILE):
        with open(ACTIVE_FILE) as f:
            name = json.load(f).get("active_profile")
        if name:
            return str(name)
    return DEFAULT_PROFILE


def load_profile(name: str | None = None) -> DetectorProfile:
    name = name or active_profile_name()
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise CalibrationError(
            f"no detector profile {name!r} in {PROFILE_DIR} "
            f"(have: {', '.join(list_profiles()) or 'none'})")
    with open(path) as f:
        doc = json.load(f)
    declared = doc.get("profile")
    if declared and declared != name:
        raise CalibrationError(
            f"{path} declares profile {declared!r} but is filed as {name!r}")
    return DetectorProfile(name=name, path=path, doc=doc)


def suffix_for(name: str) -> str:
    return f".{name}"


def assert_reportable(profile: DetectorProfile) -> None:
    problems = []
    if profile.is_preliminary:
        problems.append(f"status={profile.status!r} (not {STATUS_FINAL!r})")
    if not profile.event_level:
        problems.append("evaluation is not event-level")
    if not profile.timing_jitter_provenance.startswith("measured"):
        problems.append(
            f"timing jitter is {profile.timing_jitter_provenance!r}, not measured")
    if problems:
        raise CalibrationError(
            f"profile {profile.name!r} is not reportable: " + "; ".join(problems))






def add_profile_argument(parser) -> None:
    parser.add_argument(
        "--detector_profile", default=None,
        help=("detector calibration profile to run under (default: "
              f"${ENV_VAR}, active.json, then {DEFAULT_PROFILE}). Available: "
              f"{', '.join(list_profiles()) or 'none'}"))


def resolve(args=None, name: str | None = None,
            verbose: bool = True) -> DetectorProfile:
    if name is None:
        name = getattr(args, "detector_profile", None)
    profile = load_profile(name)
    if verbose:
        print(profile.summary_line())
        if profile.is_preliminary:
            print(f"[calib] WARNING: {profile.name} is PRELIMINARY -- outputs "
                  "are provisional and must not be quoted in the manuscript")
    return profile
