import pytest
from expecttest import assert_expected_inline

from rware.warehouse import Warehouse


def test_rendering():
    env = Warehouse(
        shelf_columns=3, shelf_rows=1, column_height=8, render_mode="rgb_array"
    )
    env.reset()
    expected_render = env.render()
    assert_expected_inline(
        str(expected_render.mean().round()), """204.0"""
    )
    assert_expected_inline(
        str(expected_render),
        """\
[[[255 255 255]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]
  [255 255 255]
  [255 255 255]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  ...
  [  0   0   0]
  [  0   0   0]
  [255 255 255]]

 [[  0   0   0]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]
  [255 255 255]
  [  0   0   0]]

 ...

 [[  0   0   0]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]
  [255 255 255]
  [  0   0   0]]

 [[  0   0   0]
  [255 255 255]
  [255 255 255]
  ...
  [255 255 255]
  [255 255 255]
  [  0   0   0]]

 [[  0   0   0]
  [  0   0   0]
  [  0   0   0]
  ...
  [  0   0   0]
  [  0   0   0]
  [  0   0   0]]]""",
    )
