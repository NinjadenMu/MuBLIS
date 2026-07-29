import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

ROW = 101
NO_TRANS = 111
UPPER = 121
NON_UNIT = 131
LEFT = 141

ROOT = Path(__file__).resolve().parents[2]

DTYPES = (
  ("s", np.float32, ctypes.c_float),
  ("d", np.float64, ctypes.c_double),
)

def library_path():
  override = os.environ.get("MUBLIS_LIB")
  if override:
    path = Path(override)
    return path if path.is_absolute() else ROOT / path

  config = os.environ.get("MUBLIS_CONFIG", "reference")

  if sys.platform == "darwin":
    suffix = ".dylib"
  elif sys.platform == "win32":
    suffix = ".dll"
  else:
    suffix = ".so"

  return ROOT / "build" / config / "lib" / f"libmublis{suffix}"

def declare_functions(lib):
  integer = ctypes.c_int
  pointer = ctypes.c_void_p

  signatures = {
    "gemm": [
      integer, integer, integer,
      integer, integer, integer,
      "scalar",
      pointer, integer,
      pointer, integer,
      "scalar",
      pointer, integer,
    ],
    "symm": [
      integer, integer, integer,
      integer, integer,
      "scalar",
      pointer, integer,
      pointer, integer,
      "scalar",
      pointer, integer,
    ],
    "syrk": [
      integer, integer, integer,
      integer, integer,
      "scalar",
      pointer, integer,
      "scalar",
      pointer, integer,
    ],
    "syr2k": [
      integer, integer, integer,
      integer, integer,
      "scalar",
      pointer, integer,
      pointer, integer,
      "scalar",
      pointer, integer,
    ],
    "trmm": [
      integer, integer, integer, integer, integer,
      integer, integer,
      "scalar",
      pointer, integer,
      pointer, integer,
    ],
    "trsm": [
      integer, integer, integer, integer, integer,
      integer, integer,
      "scalar",
      pointer, integer,
      pointer, integer,
    ],
  }

  for prefix, _, scalar in DTYPES:
    for routine, signature in signatures.items():
      function = getattr(lib, f"cblas_{prefix}{routine}")
      function.argtypes = [
        scalar if argument == "scalar" else argument
        for argument in signature
      ]
      function.restype = None

def ptr(values):
  return ctypes.c_void_p(values.ctypes.data)

def benchmark(function, arguments, flops, iterations):
  function(*arguments)

  start = time.perf_counter()

  for _ in range(iterations):
    function(*arguments)

  elapsed = time.perf_counter() - start

  return flops * iterations / elapsed / 1e9

def make_cases(lib, prefix, dtype, scalar, m, n, k):
  one = scalar(1.0)
  zero = scalar(0.0)

  a_gemm = np.ones((m, k), dtype=dtype)
  b_gemm = np.ones((k, n), dtype=dtype)
  c_gemm = np.zeros((m, n), dtype=dtype)

  a_symm = np.eye(m, dtype=dtype)
  b_symm = np.ones((m, n), dtype=dtype)
  c_symm = np.zeros((m, n), dtype=dtype)

  a_syrk = np.ones((n, k), dtype=dtype)
  c_syrk = np.zeros((n, n), dtype=dtype)

  a_syr2k = np.ones((n, k), dtype=dtype)
  b_syr2k = np.ones((n, k), dtype=dtype)
  c_syr2k = np.zeros((n, n), dtype=dtype)

  a_triangular = np.eye(m, dtype=dtype)
  b_trmm = np.ones((m, n), dtype=dtype)
  b_trsm = np.ones((m, n), dtype=dtype)

  return [
    (
      "gemm",
      getattr(lib, f"cblas_{prefix}gemm"),
      (
        ROW, NO_TRANS, NO_TRANS, m, n, k, one,
        ptr(a_gemm), k,
        ptr(b_gemm), n,
        zero,
        ptr(c_gemm), n,
      ),
      2 * m * n * k,
      (a_gemm, b_gemm, c_gemm),
    ),
    (
      "symm",
      getattr(lib, f"cblas_{prefix}symm"),
      (
        ROW, LEFT, UPPER, m, n, one,
        ptr(a_symm), m,
        ptr(b_symm), n,
        zero,
        ptr(c_symm), n,
      ),
      2 * m * m * n,
      (a_symm, b_symm, c_symm),
    ),
    (
      "syrk",
      getattr(lib, f"cblas_{prefix}syrk"),
      (
        ROW, UPPER, NO_TRANS, n, k, one,
        ptr(a_syrk), k,
        zero,
        ptr(c_syrk), n,
      ),
      n * (n + 1) * k,
      (a_syrk, c_syrk),
    ),
    (
      "syr2k",
      getattr(lib, f"cblas_{prefix}syr2k"),
      (
        ROW, UPPER, NO_TRANS, n, k, one,
        ptr(a_syr2k), k,
        ptr(b_syr2k), k,
        zero,
        ptr(c_syr2k), n,
      ),
      2 * n * (n + 1) * k,
      (a_syr2k, b_syr2k, c_syr2k),
    ),
    (
      "trmm",
      getattr(lib, f"cblas_{prefix}trmm"),
      (
        ROW, LEFT, UPPER, NO_TRANS, NON_UNIT,
        m, n, one,
        ptr(a_triangular), m,
        ptr(b_trmm), n,
      ),
      m * m * n,
      (a_triangular, b_trmm),
    ),
    (
      "trsm",
      getattr(lib, f"cblas_{prefix}trsm"),
      (
        ROW, LEFT, UPPER, NO_TRANS, NON_UNIT,
        m, n, one,
        ptr(a_triangular), m,
        ptr(b_trsm), n,
      ),
      m * m * n,
      (a_triangular, b_trsm),
    ),
  ]

def positive_integer(value):
  value = int(value)

  if value <= 0:
    raise argparse.ArgumentTypeError("value must be positive")

  return value

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("m", type=positive_integer)
  parser.add_argument("n", type=positive_integer)
  parser.add_argument("k", type=positive_integer)
  parser.add_argument("iterations", type=positive_integer)
  args = parser.parse_args()

  path = library_path()

  if not path.is_file():
    parser.error(f"MuBLIS shared library not found at {path}")

  lib = ctypes.CDLL(str(path))
  declare_functions(lib)

  for prefix, dtype, scalar in DTYPES:
    for routine, function, arguments, flops, _ in make_cases(
      lib,
      prefix,
      dtype,
      scalar,
      args.m,
      args.n,
      args.k,
    ):
      gflops = benchmark(function, arguments, flops, args.iterations)
      print(f"cblas_{prefix}{routine}: {gflops:.2f} GFLOP/s")


if __name__ == "__main__":
  main()
  