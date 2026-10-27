"""
Merge behaviour for regions spanning several overlapping sources.

These run offline against a fake store, so they exercise the merge logic
without reaching for real data.
"""
import numpy as np
import pytest

from ac_data_analysis.cutout import SourceInfo, merge_cutouts_from_sources


class FakeStore:
    """Stands in for a TensorStore, indexed as store[0, 0, z, y, x].read().result()."""

    def __init__(self, array):
        self.array = array

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        # drop the two leading channel/time indices
        self.pending = self.array[key[2:]]
        return self

    def read(self):
        return self

    def result(self):
        return self.pending


def make_source(name, array, translation):
    src = SourceInfo(name=name, url=f"memory://{name}", driver="zarr")
    src.store = FakeStore(array)
    src.translation_voxel = np.array(translation)
    src.bounds = ((0, 0, 0), array.shape)
    src.is_accessible = True
    return src


def merge(sources, shape, strategy="first_valid"):
    return merge_cutouts_from_sources(
        sources, region_start=(0, 0, 0), region_stop=shape, mip_level=0,
        merge_strategy=strategy,
    )


def test_padding_does_not_beat_real_data():
    """A tile's black edge must not win over a neighbour's real data.

    Two sources overlap on y in [8, 12). src0 is padded black over its last
    four rows; src1 has real data throughout. The seam must not go black.
    """
    a = np.full((4, 12, 4), 100, dtype=np.uint16)
    a[:, 8:, :] = 0                                    # black padding at the tile edge
    b = np.full((4, 12, 4), 200, dtype=np.uint16)      # real data across the overlap

    merged, info = merge(
        [make_source("src0", a, (0, 0, 0)), make_source("src1", b, (0, 8, 0))],
        shape=(4, 20, 4),
    )

    seam = merged[:, 8:12, :]
    assert not np.any(seam == 0), "padding won over real data at the seam"
    assert np.all(seam == 200)
    assert info["coverage_ratio"] == 1.0
    assert info["padding_voxels"] == 0


def test_first_source_still_wins_where_both_are_valid():
    """Priority among real data is unchanged: the first source listed wins."""
    a = np.full((4, 12, 4), 100, dtype=np.uint16)
    b = np.full((4, 12, 4), 200, dtype=np.uint16)

    merged, _ = merge(
        [make_source("src0", a, (0, 0, 0)), make_source("src1", b, (0, 8, 0))],
        shape=(4, 20, 4),
    )

    assert np.all(merged[:, 8:12, :] == 100)


def test_zero_is_kept_when_no_source_has_data_there():
    """Where every source is padding, the voxel stays zero and is reported."""
    a = np.zeros((4, 12, 4), dtype=np.uint16)
    b = np.zeros((4, 12, 4), dtype=np.uint16)

    merged, info = merge(
        [make_source("src0", a, (0, 0, 0)), make_source("src1", b, (0, 8, 0))],
        shape=(4, 20, 4),
    )

    assert np.all(merged == 0)
    assert info["coverage_ratio"] == 1.0, "padding still counts as covered"
    assert info["padding_voxels"] == merged.size, "but it is flagged as padding"


def test_region_beyond_every_source_is_unfilled():
    a = np.full((4, 8, 4), 100, dtype=np.uint16)

    merged, info = merge([make_source("src0", a, (0, 0, 0))], shape=(4, 16, 4))

    assert info["unfilled_voxels"] == 4 * 8 * 4
    assert info["coverage_ratio"] == pytest.approx(0.5)


def test_nan_counts_as_padding_for_float_sources():
    a = np.full((4, 12, 4), np.nan, dtype=np.float32)
    b = np.full((4, 12, 4), 7.0, dtype=np.float32)

    merged, _ = merge(
        [make_source("src0", a, (0, 0, 0)), make_source("src1", b, (0, 8, 0))],
        shape=(4, 20, 4),
    )

    assert np.all(merged[:, 8:12, :] == 7.0)


@pytest.mark.parametrize("strategy", ["average", "max", "min"])
def test_other_strategies_still_cover_the_region(strategy):
    a = np.full((4, 12, 4), 100, dtype=np.uint16)
    b = np.full((4, 12, 4), 200, dtype=np.uint16)

    _, info = merge(
        [make_source("src0", a, (0, 0, 0)), make_source("src1", b, (0, 8, 0))],
        shape=(4, 20, 4), strategy=strategy,
    )

    assert info["coverage_ratio"] == 1.0
