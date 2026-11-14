"""Run neurigami on a single-layer model and cache the answer.

neurigami needs its own Python environment (onnx, qonnx, amaranth, scipy), so
it is never imported here. ``script/neurigami_single_layer.py`` runs in that
environment as a subprocess and prints one JSON line. This module launches it,
reads the line, and caches the result in memory and on disk so a campaign pays
one build for each distinct (op, size, clock, part).

Where neurigami lives is read from the evaluator settings ``neurigami_home`` and
``neurigami_python``, or from the environment variables ``NEURIGAMI_HOME`` and
``NEURIGAMI_PYTHON``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from system.utils.UnsupportedEvaluation import ResourceExceeded, UnsupportedEvaluation

WORKER = Path(__file__).resolve().parents[2] / "script" / "neurigami_single_layer.py"
DEFAULT_CACHE_DIR = Path("output/neurigami_cache")
#: Bump when the worker's model or the meaning of its numbers changes.
CACHE_VERSION = 2
BUILD_TIMEOUT_S = 600

_MEMORY = {}


def locate(home=None, python=None):
    """Return ``(home, python)`` or raise UnsupportedEvaluation if neurigami is not set up."""
    home = home or os.environ.get("NEURIGAMI_HOME")
    if not home:
        raise UnsupportedEvaluation(
            "the neurigami evaluator needs a neurigami checkout: set the evaluator "
            "setting 'neurigami_home' or the NEURIGAMI_HOME environment variable"
        )
    if not (Path(home) / "src" / "compile.py").is_file():
        raise UnsupportedEvaluation(f"'{home}' is not a neurigami checkout")
    python = python or os.environ.get("NEURIGAMI_PYTHON") or str(
        Path(home) / ".venv" / "bin" / "python"
    )
    if not Path(python).is_file():
        python = python if python != sys.executable else None
    if not python or not Path(python).is_file():
        raise UnsupportedEvaluation(
            "the neurigami Python was not found: set 'neurigami_python' or NEURIGAMI_PYTHON"
        )
    return str(home), str(python)


def _key(op, elements, clock_ns, part, param, vector, budget):
    return (CACHE_VERSION, op, int(elements), float(clock_ns), part, param, vector,
            tuple(sorted((budget or {}).items())))


def _cache_file(cache_dir, key):
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{key[1]}_{key[2]}_{digest}.json"


def estimate_single_layer(
    op,
    elements,
    clock_ns,
    part="xczu7ev",
    param=None,
    vector=None,
    budget=None,
    home=None,
    python=None,
    cache_dir=DEFAULT_CACHE_DIR,
):
    """neurigami's estimate for one ``op`` over ``elements`` values.

    ``param`` is the slope of leaky_relu or the threshold of thresholded_relu. ``vector`` is the length of the
    axis a softmax normalises over. ``budget`` maps ``lut``, ``ff``, ``bram36``, ``dsp`` to the most the design
    may use; neurigami then picks lanes that fit it.

    Returns the worker's dict: lut, ff, bram36, dsp, uram, interval,
    first_latency, rows, clock_ns, part.
    """
    key = _key(op, elements, clock_ns, part, param, vector, budget)
    if key in _MEMORY:
        return _checked(_MEMORY[key], op, elements)
    path = _cache_file(cache_dir, key) if cache_dir else None
    if path is not None and path.is_file():
        _MEMORY[key] = json.loads(path.read_text())
        return _checked(_MEMORY[key], op, elements)

    home, python = locate(home, python)
    command = [
        python, str(WORKER), "--home", home, "--op", op,
        "--elements", str(int(elements)), "--clock-ns", repr(float(clock_ns)),
        "--part", part,
    ]
    if param is not None:
        command += ["--param", repr(float(param))]
    if vector is not None:
        command += ["--vector", str(int(vector))]
    for name, value in sorted((budget or {}).items()):
        command += [f"--budget-{name}", repr(float(value))]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as error:
        raise UnsupportedEvaluation(
            f"neurigami took more than {BUILD_TIMEOUT_S} s for {op} x {elements}"
        ) from error
    if done.returncode != 0:
        tail = " | ".join(done.stderr.strip().splitlines()[-3:])
        raise UnsupportedEvaluation(
            f"neurigami could not build {op} x {elements}: {tail or 'no error text'}"
        )
    try:
        result = json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        raise UnsupportedEvaluation(
            f"neurigami printed no result for {op} x {elements}"
        ) from error
    _MEMORY[key] = result
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result))
    return _checked(result, op, elements)


def _checked(result, op, elements):
    """The result, or ResourceExceeded where neurigami found no design that fits the budget."""
    if "no_fit" in result:
        raise ResourceExceeded(f"neurigami: {op} x {elements}: {result['no_fit']}")
    return result
