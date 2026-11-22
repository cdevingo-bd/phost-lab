"""Metrics the lab reports per cell, as data (name, unit, direction) so a UI can label and sort them."""

from __future__ import annotations

from dataclasses import dataclass

from phost import MetricSpec

# floor below which an RMS difference is rounding noise: ratios to a control that small mean nothing
NOISE_FLOOR = 1e-12

BUILTIN = (
    MetricSpec("step_ms", "ms", "lower", "timing", "median time of one step, as the driver launches it (a graph replay by default)"),
    MetricSpec("speedup", "x", "higher", "gauge", "native step_ms divided by this graph's step_ms"),
    MetricSpec("native_spread", "1", "lower", "gauge", "relative difference between the two native runs (timing noise)"),
    MetricSpec("rms_dqpos", "rad|m", "lower", "gauge", "RMS state difference to native after the run"),
    MetricSpec("control_rms_dqpos", "rad|m", "lower", "gauge", "RMS state difference native vs native (the noise floor)"),
    MetricSpec("accuracy_ratio", "1", "lower", "gauge",
               "rms_dqpos / control_rms_dqpos; null when the control is at rounding-noise level. An engine whose native "
               "steps repeat exactly is measured against a rounding floor instead (cell.accuracy_basis says so)"),
)


def accuracy_ratio(rms: float, control: float):
    """rms / control, or None when the control is rounding noise (the ratio would be meaningless)."""
    return None if control < NOISE_FLOOR else rms / control
