# ac_data_analysis

Extract 3D cutouts from large microscopy datasets described by a Neuroglancer state link.

Given a Neuroglancer link, `ac_data_analysis` resolves the zarr / precomputed sources behind
one of its layers, downloads a sub-volume centered on a point, and stitches the result
together if the requested region spans several overlapping sources.

## Installation

```sh
pip install git+https://github.com/AllenInstitute/ac_data_analysis.git
```

Or, for a local checkout with the example-notebook extras:

```sh
git clone https://github.com/AllenInstitute/ac_data_analysis.git
cd ac_data_analysis
uv sync --extra examples
```

## Quick start

This example uses a publicly readable iSPIM dataset, so it runs as-is with no credentials:

```python
from ac_data_analysis import cutout_from_NG

NG_LINK = (
    "https://neuroglancer-demo.appspot.com/#!"
    "https://apex-connects.s3.us-east-2.amazonaws.com/axonal_connectomics/"
    "contrast_adjusted/H17_PO6_S3_20250422/state2.json"
)

img, meta = cutout_from_NG(
    ng_link=NG_LINK,
    size=(128, 128, 128),   # (z, y, x) in voxels at mip 0
    mip="2",                # note: a string, not an int
    out_path="./cutouts/",
    post_fix="quickstart",
)

print(img.shape, meta["merge_info"]["coverage_ratio"])
```

This writes `{sources}_mip{mip}_{post_fix}.tif` plus a matching `_metadata.json`, and returns
the volume as an array along with the metadata dict.

The dataset behind that link is tiled into three overlapping sources, so it also exercises
the cross-source stitching described below.

### Parameters

| Parameter | Description |
|---|---|
| `ng_link` | Neuroglancer link: a full URL with `#!`, a state-JSON URL, or a raw JSON string |
| `size` | `(z, y, x)` in voxels at mip 0; divided by `2**mip` internally |
| `center` | Optional `(z, y, x)` voxel center. Defaults to the position stored in the state |
| `layer_name` | Layer to cut from (default: first layer) |
| `mip` | Resolution level, passed as a string (default `"0"`) |
| `offset` | Optional offset subtracted from the voxel center |
| `out_path`, `post_fix` | Output location and filename suffix (default suffix: timestamp) |
| `save_tif` | Write a `.tif` (default `True`) |
| `source_preference` | Source names in priority order, used when sources overlap |
| `merge_strategy` | `first_valid` (default), `average`, `max` or `min` |
| `enable_cross_source` | Stitch across overlapping sources (default `True`) |
| `debug_transforms` | Print the layer transform matrices |

### Output

`cutout_from_NG` returns `(array, metadata)`. The array is ordered **x, y, z** — transposed
from the internal z, y, x order, so it drops straight into viewers that expect xyz.

The metadata dict records `NG_link`, `center`, `size`, `mip`, the output paths, and
`merge_info` — coverage ratio, which sources contributed, and how many voxels went unfilled.
Check `merge_info` when a region sits near the edge of a dataset: a coverage ratio below 1.0
means part of the requested box had no data behind it.

`download_cutout_from_zarr(ng_link, layer_name, center, size, ...)` is a thin wrapper that
calls `cutout_from_NG` with `enable_cross_source=False`, for single-source cutouts.

### Choosing a merge strategy

When a region spans several overlapping sources, `merge_strategy` decides which value wins:

| Strategy | Behaviour |
|---|---|
| `first_valid` (default) | The first source with real data at a voxel claims it |
| `average` | Mean of the sources with real data there |
| `max` / `min` | Element-wise maximum / minimum across those sources |

Every strategy ignores background padding where another source has real data. Tiled
acquisitions pad each tile's edge with zeros, and in an overlap that padding must not beat
a neighbour's real data — otherwise seams come out as black bands, averages get dragged
down, and a minimum is always zero. Among sources that all have real data, `first_valid`
keeps the first, so `source_preference` ordering behaves as you would expect.

Where *no* source has real data the voxel is left at zero, including for float sources
padded with NaN, so padding never propagates into downstream analysis.

`merge_info` reports two separate numbers:

- `unfilled_voxels` — no source covered them at all; these lower `coverage_ratio`
- `padding_voxels` — covered, but every source offered only background there

A large `padding_voxels` means the cutout is mostly outside the imaged material, even
though `coverage_ratio` reads `1.0`.

## Command line

```sh
ac-cutout \
  --ng_link "<neuroglancer link>" \
  --size 1000 1000 1000 \
  --mip 2 \
  --out ./cutouts/
```

Optional flags: `--layer`, `--center Z Y X`, `--no-save-tif`,
`--merge-strategy {first_valid,average,max,min}`.

The same CLI is available as `python -m ac_data_analysis.cutout`.

## TLS certificates

TLS certificates are verified. If your data is served behind an internal certificate
authority, Python needs to trust that CA — but a typical cutout touches **two** hosts: the
state server (often the publicly trusted `json.neurodata.io`) and the data server (often
internal). Pointing at your internal CA alone will break the public host.

So use a bundle containing *both*. Append your internal CA to the public roots:

```sh
python -c "import certifi, shutil, sys; shutil.copy(certifi.where(), 'combined-ca.pem')"
cat /path/to/rootCA.pem >> combined-ca.pem

export REQUESTS_CA_BUNDLE=$PWD/combined-ca.pem
export CURL_CA_BUNDLE=$PWD/combined-ca.pem   # also picked up by tensorstore
export SSL_CERT_FILE=$PWD/combined-ca.pem
```

Symptoms when this is wrong:

| Error | Cause |
|---|---|
| `failed to initialize` for every source | tensorstore can't verify the data host — `CURL_CA_BUNDLE` missing or incomplete |
| `CERTIFICATE_VERIFY_FAILED` on `json.neurodata.io` | bundle has your internal CA but not the public roots |

As a last resort, `AC_CUTOUT_VERIFY_SSL=0` disables verification for the `requests` calls.
It does not affect tensorstore, so it will not fix `failed to initialize` on its own.

## Known limitations

- Visor zarr sources are not currently supported.
- `mip` is a string, not an integer — `mip="2"`, not `mip=2`.

## Examples

`examples/example_cutout.ipynb` runs a cutout end to end and views the result with
`microviewer` and matplotlib.

## License

Distributed under the Allen Institute Software License (2-clause BSD plus a clause
prohibiting commercial redistribution without permission). See [LICENSE](LICENSE).
For commercial licensing, contact terms@alleninstitute.org.
