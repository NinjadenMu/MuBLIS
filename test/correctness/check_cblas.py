import ctypes
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import blas

ROW = 101
COL = 102
NO_TRANS = 111
TRANS = 112
UPPER = 121
LOWER = 122
NON_UNIT = 131
UNIT = 132
LEFT = 141
RIGHT = 142

ROOT = Path(__file__).resolve().parents[2]

DTYPES = {
  np.dtype(np.float32): {
    "prefix": "s",
    "ctype": ctypes.c_float,
    "rtol": 5e-4,
    "atol": 5e-5,
  },
  np.dtype(np.float64): {
    "prefix": "d",
    "ctype": ctypes.c_double,
    "rtol": 2e-11,
    "atol": 1e-12,
  },
}

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

  for info in DTYPES.values():
    scalar = info["ctype"]
    prefix = info["prefix"]

    for routine, signature in signatures.items():
      argtypes = [
        scalar if argument == "scalar" else argument
        for argument in signature
      ]

      function = getattr(lib, f"cblas_{prefix}{routine}")
      function.argtypes = argtypes
      function.restype = None

@pytest.fixture(scope="session")
def mublis():
  path = library_path()

  if not path.is_file():
    pytest.fail(
      f"MuBLIS shared library not found at {path}.",
      pytrace=False,
    )

  lib = ctypes.CDLL(str(path))
  declare_functions(lib)
  return lib

@pytest.fixture(
  params=[np.dtype(np.float32), np.dtype(np.float64)],
  ids=["float32", "float64"],
)
def dtype(request):
  return request.param

@pytest.fixture(
  params=[("C", ROW), ("F", COL)],
  ids=["row-major", "column-major"],
)
def layout(request):
  return request.param

@pytest.fixture
def rng():
  return np.random.default_rng(314159)

def make_array(values, dtype, order):
  return np.array(values, dtype=dtype, order=order, copy=True)

def random_matrix(rng, shape, dtype, order):
  return make_array(rng.standard_normal(shape), dtype, order)

def fortran_copy(values, dtype):
  return np.array(values, dtype=dtype, order="F", copy=True)

def ptr(values):
  return ctypes.c_void_p(values.ctypes.data)

def leading_dimension(values, order):
  axis = 0 if order == "C" else 1
  return values.strides[axis] // values.itemsize

def cblas_function(lib, dtype, routine):
  prefix = DTYPES[dtype]["prefix"]
  return getattr(lib, f"cblas_{prefix}{routine}")

def scipy_function(dtype, routine):
  return blas.get_blas_funcs(routine, dtype=dtype)

def assert_close(actual, expected, dtype):
  info = DTYPES[dtype]

  np.testing.assert_allclose(
    actual,
    expected,
    rtol=info["rtol"],
    atol=info["atol"],
  )

def symmetric_storage(rng, size, dtype, order, lower):
  values = rng.standard_normal((size, size))

  if lower:
    values[np.triu_indices(size, 1)] = np.nan
  else:
    values[np.tril_indices(size, -1)] = np.nan

  return make_array(values, dtype, order)

def triangular_storage(
  rng,
  size,
  dtype,
  order,
  lower,
  unit_diagonal,
):
  values = rng.uniform(-0.25, 0.25, size=(size, size))

  if lower:
    values[np.triu_indices(size, 1)] = np.nan
  else:
    values[np.tril_indices(size, -1)] = np.nan

  if unit_diagonal:
    np.fill_diagonal(values, np.nan)
  else:
    np.fill_diagonal(values, rng.uniform(1.5, 2.5, size=size))

  return make_array(values, dtype, order)

@pytest.mark.parametrize(
  "trans_a,trans_b",
  list(itertools.product((False, True), repeat=2)),
)
def test_gemm(mublis, rng, dtype, layout, trans_a, trans_b):
  order, cblas_order = layout
  m, n, k = 5, 4, 3
  alpha, beta = 1.25, -0.5

  a_shape = (k, m) if trans_a else (m, k)
  b_shape = (n, k) if trans_b else (k, n)

  a = random_matrix(rng, a_shape, dtype, order)
  b = random_matrix(rng, b_shape, dtype, order)
  initial_c = random_matrix(rng, (m, n), dtype, order)
  actual = make_array(initial_c, dtype, order)

  expected = scipy_function(dtype, "gemm")(
    alpha,
    fortran_copy(a, dtype),
    fortran_copy(b, dtype),
    beta=beta,
    c=fortran_copy(initial_c, dtype),
    trans_a=int(trans_a),
    trans_b=int(trans_b),
    overwrite_c=1,
  )

  cblas_function(mublis, dtype, "gemm")(
    cblas_order,
    TRANS if trans_a else NO_TRANS,
    TRANS if trans_b else NO_TRANS,
    m,
    n,
    k,
    alpha,
    ptr(a),
    leading_dimension(a, order),
    ptr(b),
    leading_dimension(b, order),
    beta,
    ptr(actual),
    leading_dimension(actual, order),
  )

  assert_close(actual, expected, dtype)

@pytest.mark.parametrize("right", [False, True], ids=["left", "right"])
@pytest.mark.parametrize("lower", [False, True], ids=["upper", "lower"])
def test_symm(mublis, rng, dtype, layout, right, lower):
  order, cblas_order = layout
  m, n = 5, 3
  alpha, beta = 0.75, -0.25
  symmetric_size = n if right else m

  a = symmetric_storage(
    rng,
    symmetric_size,
    dtype,
    order,
    lower,
  )
  b = random_matrix(rng, (m, n), dtype, order)
  initial_c = random_matrix(rng, (m, n), dtype, order)
  actual = make_array(initial_c, dtype, order)

  expected = scipy_function(dtype, "symm")(
    alpha,
    fortran_copy(a, dtype),
    fortran_copy(b, dtype),
    beta=beta,
    c=fortran_copy(initial_c, dtype),
    side=int(right),
    lower=int(lower),
    overwrite_c=1,
  )

  cblas_function(mublis, dtype, "symm")(
    cblas_order,
    RIGHT if right else LEFT,
    LOWER if lower else UPPER,
    m,
    n,
    alpha,
    ptr(a),
    leading_dimension(a, order),
    ptr(b),
    leading_dimension(b, order),
    beta,
    ptr(actual),
    leading_dimension(actual, order),
  )

  assert_close(actual, expected, dtype)

@pytest.mark.parametrize("trans", [False, True], ids=["no-trans", "trans"])
@pytest.mark.parametrize("lower", [False, True], ids=["upper", "lower"])
def test_syrk(mublis, rng, dtype, layout, trans, lower):
  order, cblas_order = layout
  n, k = 5, 3
  alpha, beta = 1.25, -0.5

  a_shape = (k, n) if trans else (n, k)
  a = random_matrix(rng, a_shape, dtype, order)
  initial_c = random_matrix(rng, (n, n), dtype, order)
  actual = make_array(initial_c, dtype, order)

  expected = scipy_function(dtype, "syrk")(
    alpha,
    fortran_copy(a, dtype),
    beta=beta,
    c=fortran_copy(initial_c, dtype),
    trans=int(trans),
    lower=int(lower),
    overwrite_c=1,
  )

  cblas_function(mublis, dtype, "syrk")(
    cblas_order,
    LOWER if lower else UPPER,
    TRANS if trans else NO_TRANS,
    n,
    k,
    alpha,
    ptr(a),
    leading_dimension(a, order),
    beta,
    ptr(actual),
    leading_dimension(actual, order),
  )

  active = np.tril_indices(n) if lower else np.triu_indices(n)
  inactive = (
    np.triu_indices(n, 1)
    if lower
    else np.tril_indices(n, -1)
  )

  assert_close(actual[active], expected[active], dtype)
  np.testing.assert_array_equal(actual[inactive], initial_c[inactive])

@pytest.mark.parametrize("trans", [False, True], ids=["no-trans", "trans"])
@pytest.mark.parametrize("lower", [False, True], ids=["upper", "lower"])
def test_syr2k(mublis, rng, dtype, layout, trans, lower):
  order, cblas_order = layout
  n, k = 5, 3
  alpha, beta = 0.75, -0.25

  operand_shape = (k, n) if trans else (n, k)
  a = random_matrix(rng, operand_shape, dtype, order)
  b = random_matrix(rng, operand_shape, dtype, order)
  initial_c = random_matrix(rng, (n, n), dtype, order)
  actual = make_array(initial_c, dtype, order)

  expected = scipy_function(dtype, "syr2k")(
    alpha,
    fortran_copy(a, dtype),
    fortran_copy(b, dtype),
    beta=beta,
    c=fortran_copy(initial_c, dtype),
    trans=int(trans),
    lower=int(lower),
    overwrite_c=1,
  )

  cblas_function(mublis, dtype, "syr2k")(
    cblas_order,
    LOWER if lower else UPPER,
    TRANS if trans else NO_TRANS,
    n,
    k,
    alpha,
    ptr(a),
    leading_dimension(a, order),
    ptr(b),
    leading_dimension(b, order),
    beta,
    ptr(actual),
    leading_dimension(actual, order),
  )

  active = np.tril_indices(n) if lower else np.triu_indices(n)
  inactive = (
    np.triu_indices(n, 1)
    if lower
    else np.tril_indices(n, -1)
  )

  assert_close(actual[active], expected[active], dtype)
  np.testing.assert_array_equal(actual[inactive], initial_c[inactive])

@pytest.mark.parametrize("routine", ["trmm", "trsm"])
@pytest.mark.parametrize(
  "right,lower,trans,unit_diagonal",
  list(itertools.product((False, True), repeat=4)),
)
def test_triangular_routine(
  mublis,
  rng,
  dtype,
  layout,
  routine,
  right,
  lower,
  trans,
  unit_diagonal,
):
  order, cblas_order = layout
  m, n = 5, 3
  alpha = 0.75
  triangular_size = n if right else m

  a = triangular_storage(
    rng,
    triangular_size,
    dtype,
    order,
    lower,
    unit_diagonal,
  )
  initial_b = random_matrix(rng, (m, n), dtype, order)
  actual = make_array(initial_b, dtype, order)

  expected = scipy_function(dtype, routine)(
    alpha,
    fortran_copy(a, dtype),
    fortran_copy(initial_b, dtype),
    side=int(right),
    lower=int(lower),
    trans_a=int(trans),
    diag=int(unit_diagonal),
    overwrite_b=1,
  )

  cblas_function(mublis, dtype, routine)(
    cblas_order,
    RIGHT if right else LEFT,
    LOWER if lower else UPPER,
    TRANS if trans else NO_TRANS,
    UNIT if unit_diagonal else NON_UNIT,
    m,
    n,
    alpha,
    ptr(a),
    leading_dimension(a, order),
    ptr(actual),
    leading_dimension(actual, order),
  )

  assert_close(actual, expected, dtype)
  