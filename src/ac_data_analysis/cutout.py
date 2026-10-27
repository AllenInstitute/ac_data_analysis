#!/usr/bin/env python3
"""
Download a cutout spanning one or more precomputed/zarr sources described in a
Neuroglancer state link.

Dependencies:
  - tensorstore
  - numpy
  - requests
  - tifffile
  - scipy

TLS note:
  TLS certificates are verified by default. A cutout usually touches two hosts -
  the state server and the data server - so a CA bundle holding only an internal
  CA will break whichever of the two is publicly trusted. Use a bundle containing
  both the public roots and your internal CA, and point all three variables at it:
  REQUESTS_CA_BUNDLE, CURL_CA_BUNDLE (tensorstore) and SSL_CERT_FILE. See the
  README for the full recipe.

  AC_CUTOUT_VERIFY_SSL=0 disables verification for the requests calls only.
"""

from __future__ import annotations
import os
import json
import logging
import datetime
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import requests
import tensorstore as ts
import tifffile as tiff
from urllib.parse import urljoin
from scipy.ndimage import affine_transform

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _verify_ssl() -> bool:
    """Whether to verify TLS certificates. See the TLS note in the module docstring."""
    return os.environ.get("AC_CUTOUT_VERIFY_SSL", "1").lower() not in ("0", "false", "no")


@dataclass
class SourceInfo:
    """Information about a data source"""
    name: str
    url: str
    driver: str = "zarr"
    bounds: Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = None
    store: Optional[ts.TensorStore] = None
    translation_voxel: Optional[np.ndarray] = None
    transform_matrix: Optional[np.ndarray] = None  # Add this
    is_accessible: bool = False


# -----------------------
# Utility parsing helpers
# -----------------------
def get_json_state_from_ng_link(ng_link: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    If ng_link is a full url to a JSON state (or contains a '#!' fragment) fetch and return
    the JSON. Accepts either a direct URL or the '#!' fragment.
    """
    if "#!" in ng_link:
        ng_link = ng_link.split("#!", 1)[1]
    # If looks like a raw json blob already, try parsing
    try:
        return json.loads(ng_link)
    except Exception:
        pass

    resp = requests.get(ng_link, timeout=timeout, verify=_verify_ssl())
    resp.raise_for_status()
    return resp.json()


def parse_source_url(url: str) -> Tuple[str, str]:
    """
    Normalize a source string that may include a driver suffix separated by '|' or be space
    separated. Returns (clean_url, driver).
    Examples:
      "https://...|zarr:..." -> ("https://...", "zarr")
      "zarr://s3://path/to/store" -> ("s3://path/to/store", "zarr")
    """
    url = url.strip()
    driver = "zarr"
    
    # Handle zarr:// scheme - strip it and use underlying URL
    if url.startswith("zarr://"):
        url = url[7:]  # Remove "zarr://" prefix
        driver = "zarr2"  # Default to zarr2 for zarr:// URLs
    
    if "|" in url:
        left, right = url.split("|", 1)
        url = left.strip()
        right = right.strip()
        # right might be "zarr:..." or plain "zarr"
        driver = right.split(":", 1)[0] if right else driver
    elif " " in url:
        url = url.split(" ", 1)[0]
    
    return url, driver


# -----------------------
# Layer/source extraction
# -----------------------
def _extract_sources_from_layer_dict(layer: Dict[str, Any]) -> List[SourceInfo]:
    """Extract SourceInfo objects from a single layer dict (Neuroglancer layer entry)."""
    sources: List[SourceInfo] = []
    src = layer.get("source")
    if isinstance(src, str):
        url_clean, driver = parse_source_url(src)
        sources.append(SourceInfo(name=layer.get("name", "default"), url=url_clean, driver=driver))
    elif isinstance(src, dict):
        # Either a dict containing 'url' or a dict of named sources
        if "url" in src:
            url_clean, driver = parse_source_url(src["url"])
            name = src.get("name", layer.get("name", "default"))
            sources.append(SourceInfo(name=name, url=url_clean, driver=driver))
        else:
            for key, value in src.items():
                if isinstance(value, str):
                    if "http" in value or value.startswith("zarr://"):
                        url_clean, driver = parse_source_url(value)
                        sources.append(SourceInfo(name=key, url=url_clean, driver=driver))
                elif isinstance(value, dict) and "url" in value:
                    url_clean, driver = parse_source_url(value["url"])
                    name = value.get("name", key)
                    sources.append(SourceInfo(name=name, url=url_clean, driver=driver))
    elif isinstance(src, list):
        for i, s in enumerate(src):
            if isinstance(s, str):
                url_clean, driver = parse_source_url(s)
                sources.append(SourceInfo(name=f"{layer.get('name','layer')}_src{i}", url=url_clean, driver=driver))
            elif isinstance(s, dict) and "url" in s:
                url_clean, driver = parse_source_url(s["url"])
                name = s.get("name", f"{layer.get('name','layer')}_src{i}")
                source_info = SourceInfo(name=name, url=url_clean, driver=driver)
    
                # Extract transform matrix if present
                if "transform" in s and "matrix" in s["transform"]:
                    source_info.transform_matrix = np.array(s["transform"]["matrix"])
                
                sources.append(source_info)
    return sources


def get_layer_sources(ng_state: Dict[str, Any], layer_name: str) -> List[SourceInfo]:
    """Find the layer by name in a Neuroglancer state and return its sources."""
    for layer in ng_state.get("layers", []):
        if layer.get("name") == layer_name:
            return _extract_sources_from_layer_dict(layer)
    raise ValueError(f"Layer with name '{layer_name}' not found")


# -----------------------
# Zarr / translation helpers
# -----------------------
def _fetch_zarr_attrs(zarr_url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Read zarr.json or .zattrs from either HTTP or local path. Return parsed JSON attributes.
    """
    if zarr_url.startswith("http"):
        # Try zarr.json then .zattrs
        for candidate in ("zarr.json", ".zattrs"):
            json_url = urljoin(zarr_url.rstrip("/") + "/", candidate)
            try:
                resp = requests.get(json_url, timeout=timeout, verify=_verify_ssl())
                if resp.status_code == 200:
                    return resp.json()
            except Exception:
                continue
        raise RuntimeError(f"Failed to fetch zarr attributes from {zarr_url}")
    else:
        for candidate in ("zarr.json", ".zattrs"):
            p = os.path.join(zarr_url, candidate)
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f)
        raise RuntimeError(f"Failed to open local zarr attrs at {zarr_url}")


def get_layer_translation_voxel(zarr_url: str) -> np.ndarray:
    """
    Compute translation (in voxels) for a zarr store by reading multiscales attributes.
    Returns integer voxel translation (z,y,x).
    """
    zattrs = _fetch_zarr_attrs(zarr_url)
    multiscale = None
    if "attributes" in zattrs and "multiscales" in zattrs["attributes"]:
        multiscale = zattrs["attributes"]["multiscales"][0]
    elif "multiscales" in zattrs:
        multiscale = zattrs["multiscales"][0]
    else:
        multiscale = {}

    # Determine spatial scale (defaults to 1.0)
    spatial_scale = np.array([1.0, 1.0, 1.0])
    # Look inside datasets -> coordinateTransformations for scale
    ds = multiscale.get("datasets", []) or []
    if ds:
        first_ds = ds[0]
        for t in first_ds.get("coordinateTransformations", []):
            if t.get("type") == "scale" and "scale" in t:
                scale = t["scale"]
                if len(scale) >= 3:
                    spatial_scale = np.array(scale[-3:])
                break

    # global translation defined at multiscale level (optional)
    translation_mip0 = [0.0, 0.0, 0.0]
    for t in multiscale.get("coordinateTransformations", []):
        if t.get("type") == "translation" and "translation" in t:
            translation_mip0 = list(t.get("translation", [0.0, 0.0, 0.0])[-3:])
            break

    translation_voxel = np.array(translation_mip0) / spatial_scale
    return translation_voxel.astype(int)

def extract_translation_from_matrix(matrix: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract translation vector from various transform matrix formats.
    Always uses last column, last 3 rows as translation vector.
    
    Returns: np.array([z, y, x]) translation or None if not found
    """
    matrix = np.array(matrix)
    
    # Check if matrix has at least 3 rows and 4+ columns
    if len(matrix.shape) == 2 and matrix.shape[0] >= 3 and matrix.shape[1] >= 4:
        # Always use last column, last 3 rows for translation
        translation = matrix[-3:, -1]  # Last 3 rows, last column
        return np.array([translation[0], translation[1], translation[2]])  # z,y,x
    else:
        logger.warning("Unsupported transform matrix shape: %s", matrix.shape)
        return None

def get_layer_translation_voxel_safe(zarr_url: str) -> np.ndarray:
    """Safe wrapper for zarr translation extraction with fallback."""
    try:
        return get_layer_translation_voxel(zarr_url)
    except Exception as e:
        logger.debug("Could not extract translation from zarr metadata: %s", e)
        return np.array([0, 0, 0], dtype=int)

def analyze_transform_matrices(sources: List[SourceInfo]) -> None:
    """Analyze all transform matrices to understand their structure."""
    logger.info("=== Transform Matrix Analysis ===")
    
    for i, src in enumerate(sources):
        logger.info(f"Source {i}: {src.name}")
        
        if src.transform_matrix is not None:
            matrix = np.array(src.transform_matrix)
            logger.info(f"  Matrix shape: {matrix.shape}")
            logger.info(f"  Matrix:\n{matrix}")
            
            # Try to extract translation from different positions
            potential_translations = {}
            
            # Check last column
            if matrix.shape[1] >= 4:
                potential_translations['last_column'] = matrix[:3, -1] if matrix.shape[0] >= 3 else None
            
            # Check specific positions (like your Y-offset case)
            if matrix.shape == (5, 6):
                potential_translations['position_[3,5]'] = matrix[3, 5]
                potential_translations['last_column_5x6'] = matrix[:3, 5]
            
            # Check for non-zero values that might be translations
            if matrix.shape[0] >= 4 and matrix.shape[1] >= 6:
                row_3 = matrix[3, :]
                non_zero_positions = np.where(row_3 != 0)[0]
                if len(non_zero_positions) > 0:
                    potential_translations['row_3_nonzero'] = {pos: row_3[pos] for pos in non_zero_positions}
            
            logger.info(f"  Potential translations: {potential_translations}")
            
            # Use the robust extraction function
            extracted = extract_translation_from_matrix(matrix)
            logger.info(f"  Extracted translation: {extracted}")
        else:
            logger.info("  No transform matrix")
        
        logger.info(f"  Final translation_voxel: {src.translation_voxel}")
        logger.info("")

# -----------------------
# TensorStore initialization
# -----------------------
def initialize_source(source: SourceInfo, mip: str = "0") -> bool:
    """
    Try opening the source with TensorStore and infer bounds (in voxel coords)
    and translation_voxel. Returns True if accessible.
    """
    try:
        tried = False
        # Try zarr2 first for .zarr URLs, then fallback to others
        drivers_to_try = [source.driver] if source.driver else []
        
        # For .zarr URLs, prioritize zarr2
        if source.url.endswith('.zarr/'):
            for d in ("zarr2", "zarr", "zarr3"):  # Changed order: zarr2 first
                if d not in drivers_to_try:
                    drivers_to_try.append(d)
        else:
            for d in ("zarr3", "zarr2", "zarr"):
                if d not in drivers_to_try:
                    drivers_to_try.append(d)

        for drv in drivers_to_try:
            try:
                spec = {
                    "driver": drv,
                    "kvstore": source.url,
                    "path": mip,
                }
                source.store = ts.open(spec).result()
                logger.debug("Successfully opened %s with driver %s", source.name, drv)
                tried = True
                break
            except Exception as e:
                logger.debug("Failed to open %s with driver %s: %s", source.name, drv, e)
                continue

        if not tried or source.store is None:
            logger.debug("Could not open TensorStore for %s using drivers %s", source.url, drivers_to_try)
            source.is_accessible = False
            return False

        # Extract domain bounds. TensorStore domain uses (time?, channel?, z, y, x) ordering typically.
        domain = source.store.domain
        # domain.inclusive_min/exclusive_max may be sequences -> convert to ints and pick last 3 dims
        inc_min = tuple(int(x) for x in domain.inclusive_min)
        exc_max = tuple(int(x) for x in domain.exclusive_max)
        if len(inc_min) < 3 or len(exc_max) < 3:
            raise RuntimeError("Unexpected domain dimensionality")
        # last 3 elements considered z,y,x
        zyx_min = tuple(inc_min[-3:])
        zyx_max = tuple(int(x - 1) for x in exc_max[-3:])
        source.bounds = (zyx_min, zyx_max)

        # Get translation from NG transform matrix if available
        if source.transform_matrix is not None:
            matrix = source.transform_matrix
            translation = extract_translation_from_matrix(matrix)
            if translation is not None:
                source.translation_voxel = translation
                logger.debug("Extracted translation %s from transform matrix for %s", translation, source.name)
            else:
                # Fallback to zarr metadata
                source.translation_voxel = get_layer_translation_voxel_safe(source.url)
        else:
            # Fallback to zarr metadata translation
            source.translation_voxel = get_layer_translation_voxel_safe(source.url)

        source.is_accessible = True
        return True

    except Exception as exc:
        logger.warning("Failed to initialize source %s: %s", source.name, exc)
        source.is_accessible = False
        return False

def debug_source_transforms(sources):
    for i, src in enumerate(sources):
        print(f"Source {i}: {src.name}")
        print(f"  Transform matrix: {src.transform_matrix}")
        print(f"  Translation voxel: {src.translation_voxel}")
        if src.transform_matrix is not None:
            matrix = src.transform_matrix
            if len(matrix) > 3 and len(matrix[3]) > 5:
                print(f"  Y offset from matrix[3][5]: {matrix[3][5]}")

# -----------------------
# Region/source ops
# -----------------------
def get_source_bounds_in_global_coords(source: SourceInfo, mip_level: int = 0) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Return (global_min, global_max) for the source at given MIP level."""
    if not source.bounds or source.translation_voxel is None:
        raise ValueError(f"Source {source.name} not initialized")
    local_min, local_max = source.bounds
    translation = source.translation_voxel // (2 ** mip_level)
    global_min = tuple(int(l + t) for l, t in zip(local_min, translation))
    global_max = tuple(int(l + t) for l, t in zip(local_max, translation))
    return global_min, global_max


def find_sources_for_region(sources: List[SourceInfo],
                            region_start: Tuple[int, int, int],
                            region_stop: Tuple[int, int, int],
                            mip_level: int = 0) -> List[SourceInfo]:
    """Return the subset of sources that overlap the region."""
    overlapping = []
    for s in sources:
        if not s.is_accessible:
            continue
        try:
            gmin, gmax = get_source_bounds_in_global_coords(s, mip_level)
            overlap = all(max(region_start[i], gmin[i]) <= min(region_stop[i], gmax[i]) for i in range(3))
            if overlap:
                overlapping.append(s)
                logger.debug("Source %s overlaps region: %s - %s", s.name, gmin, gmax)
        except Exception as exc:
            logger.warning("Could not check bounds for %s: %s", s.name, exc)
    return overlapping


# -----------------------
# Merging logic
# -----------------------
def merge_cutouts_from_sources(sources: List[SourceInfo],
                               region_start: Tuple[int, int, int],
                               region_stop: Tuple[int, int, int],
                               mip_level: int = 0,
                               merge_strategy: str = "first_valid") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read overlapping pieces from multiple sources and merge them together.
    merge_strategy: 'first_valid', 'average', 'max', 'min'
    """
    region_shape = tuple(stop - start for start, stop in zip(region_start, region_stop))
    merged_cutout = np.zeros(region_shape, dtype=np.float32)
    weight_map = np.zeros(region_shape, dtype=np.float32)
    source_map = np.full(region_shape, -1, dtype=np.int32)

    merge_info: Dict[str, Any] = {"sources_used": [], "overlap_regions": [], "merge_strategy": merge_strategy}

    for src_idx, src in enumerate(sources):
        if src.store is None:
            continue
        try:
            gmin, gmax = get_source_bounds_in_global_coords(src, mip_level)
            # compute overlap (global)
            overlap_start = tuple(max(region_start[i], gmin[i]) for i in range(3))
            overlap_stop = tuple(min(region_stop[i], gmax[i]) for i in range(3))
            if any(overlap_start[i] >= overlap_stop[i] for i in range(3)):
                continue
            # convert to local (source) voxel coords
            translation = src.translation_voxel // (2 ** mip_level)
            local_start = tuple(int(overlap_start[i] - translation[i]) for i in range(3))
            local_stop = tuple(int(overlap_stop[i] - translation[i]) for i in range(3))

            logger.debug("Reading %s local %s:%s -> global %s:%s", src.name, local_start, local_stop, overlap_start, overlap_stop)

            # Read from TensorStore. Assumes store indexing ordering [..., z, y, x]
            cutout = src.store[0, 0,
                               local_start[0]:local_stop[0],
                               local_start[1]:local_stop[1],
                               local_start[2]:local_stop[2]].read().result()
            # Ensure numpy array
            cutout = np.asarray(cutout)

            merge_start = tuple(overlap_start[i] - region_start[i] for i in range(3))
            merge_stop = tuple(merge_start[i] + cutout.shape[i] for i in range(3))
            merge_slice = tuple(slice(merge_start[i], merge_stop[i]) for i in range(3))
            current_weights = weight_map[merge_slice]

            if merge_strategy == "first_valid":
                mask = current_weights == 0
                # careful with broadcasting shapes
                merged_cutout[merge_slice][mask] = cutout[mask]
                weight_map[merge_slice][mask] = 1.0
                source_map[merge_slice][mask] = src_idx

            elif merge_strategy == "average":
                # incremental average
                merged_cutout[merge_slice] = (merged_cutout[merge_slice] * current_weights + cutout) / (current_weights + 1.0)
                weight_map[merge_slice] += 1.0
                new_mask = current_weights == 0
                source_map[merge_slice][new_mask] = src_idx

            elif merge_strategy == "max":
                no_data_mask = current_weights == 0
                new_data_mask = cutout > merged_cutout[merge_slice]
                update_mask = new_data_mask | no_data_mask
                merged_cutout[merge_slice][update_mask] = cutout[update_mask]
                weight_map[merge_slice][no_data_mask] = 1.0
                source_map[merge_slice][update_mask] = src_idx

            elif merge_strategy == "min":
                no_data_mask = current_weights == 0
                new_data_mask = (cutout < merged_cutout[merge_slice]) | no_data_mask
                merged_cutout[merge_slice][new_data_mask] = cutout[new_data_mask]
                weight_map[merge_slice][no_data_mask] = 1.0
                source_map[merge_slice][new_data_mask] = src_idx

            else:
                raise ValueError(f"Unknown merge_strategy: {merge_strategy}")

            merge_info["sources_used"].append({
                "source_name": src.name,
                "source_idx": src_idx,
                "local_region": (local_start, local_stop),
                "global_region": (overlap_start, overlap_stop),
                "cutout_shape": cutout.shape
            })
        except Exception as exc:
            logger.warning("Failed to read from source %s: %s", src.name, exc)
            continue

    unfilled_voxels = int(np.sum(weight_map == 0))
    total_voxels = int(np.prod(region_shape))
    coverage_ratio = 1.0 - (unfilled_voxels / total_voxels) if total_voxels > 0 else 0.0

    merge_info.update({
        "unfilled_voxels": unfilled_voxels,
        "total_voxels": total_voxels,
        "coverage_ratio": coverage_ratio
    })

    # choose dtype from first available source if present
    out_dtype = np.uint8
    for s in sources:
        if s.store is not None:
            try:
                out_dtype = np.dtype(s.store.dtype)
                break
            except Exception:
                continue

    return merged_cutout.astype(out_dtype), merge_info


# -----------------------
# Neuroglancer coordinates
# -----------------------
def ng_to_voxel_coord(ng_coord: Union[List[float], Tuple[float, float, float]],
                      ng_state: Dict[str, Any],
                      layer_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert Neuroglancer physical coordinate to voxel coordinate for a given layer.
    Returns (voxel_coord (z,y,x), transform_matrix or None).
    Supports a 4x4 affine matrix (matrix) or no transform.
    """
    # find layer
    layer = None
    for lyr in ng_state.get("layers", []):
        if lyr.get("name") == layer_name:
            layer = lyr
            break
    if layer is None:
        raise ValueError(f"Layer {layer_name} not found")

    # look for transform matrix (could be in layer['source']['transform'] or layer['transform'])
    transform = None
    src = layer.get("source")
    if isinstance(src, dict) and "transform" in src:
        transform = src["transform"]
    elif "transform" in layer:
        transform = layer["transform"]

    if transform and "matrix" in transform:
        mat = np.array(transform["matrix"])
        # Expect 4x4 matrix that maps voxel coordinates -> world. We invert to get world->voxel
        if mat.shape == (4, 4):
            try:
                inv = np.linalg.inv(mat)
            except np.linalg.LinAlgError:
                raise RuntimeError("Transform matrix is singular")
            # ng_coord is physical/world (x,y,z?) — consistently we'll treat last 3 entries as z,y,x order expected by this code.
            # Convert to homogeneous (x,y,z,1). Note: Neuroglancer matrix ordering varies; here we assume standard.
            x, y, z = ng_coord
            world = np.array([x, y, z, 1.0])
            voxel_h = inv.dot(world)
            voxel = np.asarray([voxel_h[2], voxel_h[1], voxel_h[0]])  # (z,y,x)
            return voxel, mat
        elif mat.shape == (5, 6):
            # Handle 5x6 Neuroglancer transform matrix
            # Format: rows=[t, c, z, y, x], columns=[t, c, z, y, x, translation]
            # Extract spatial (z,y,x) subsection: rows 2-4, columns 2-4
            # Translation is column 5 (last), rows 2-4
            logger.debug("Processing 5x6 transform matrix")
            rotation_scale = mat[2:5, 2:5]  # 3x3 spatial rotation/scale
            translation = mat[2:5, 5]       # 3x1 translation (z, y, x)
            
            try:
                inv_rot = np.linalg.inv(rotation_scale)
            except np.linalg.LinAlgError:
                logger.warning("Transform matrix rotation part is singular, using identity")
                return np.array(ng_coord), mat
            
            # Transform: world_coord = rotation_scale @ voxel_coord + translation
            # So: voxel_coord = inv_rot @ (world_coord - translation)
            x, y, z = ng_coord
            world = np.array([x, y, z])
            voxel = inv_rot @ (world - translation)
            # Reorder to z,y,x
            # voxel = np.asarray([voxel[2], voxel[1], voxel[0]])
            return voxel, mat
        else:
            logger.debug("Unsupported transform matrix shape %s; ignoring", mat.shape)
            return np.array(ng_coord), None
    else:
        return np.array(ng_coord), None


# -----------------------
# Main cross-source download
# -----------------------
def cutout_from_NG(
    ng_link: str,
    size: Union[List[int], Tuple[int, int, int]],
    center: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    out_path: Optional[str] = None,
    post_fix: Optional[str] = None,
    mip: str = "0",
    layer_name: Optional[str] = None,
    offset: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    save_tif: bool = True,
    source_preference: Optional[List[str]] = None,
    merge_strategy: str = "first_valid",
    enable_cross_source: bool = True,
    debug_transforms: bool = False 
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Main function with optional transform debugging."""
    
    logger.info("Starting cross-source cutout with merge strategy: %s", merge_strategy)

    ng_state = get_json_state_from_ng_link(ng_link)
    if layer_name is None:
        # get first layer name
        layer_name = ng_state['layers'][0]['name']
    # Robustly extract center from ng_state['position'] using dimensions order if present
    if center is None:
        pos = ng_state['position']
        if 'dimensions' in ng_state:
            # Use the order of keys in dimensions to map position indices
            dim_keys = list(ng_state['dimensions'].keys())
            idx = []
            for ax in ['z', 'y', 'x']:
                if ax in dim_keys:
                    idx.append(dim_keys.index(ax))
                else:
                    raise ValueError(f"Axis '{ax}' not found in ng_state['dimensions'] keys: {dim_keys}")
            center = [pos[i] for i in idx]
            logger.info(f"Extracted center from ng_state['position'] using dimensions order: {center} (axes order: {dim_keys})")
        elif 'names' in ng_state:
            names = ng_state['names']
            idx = []
            for ax in ['z', 'y', 'x']:
                if ax in names:
                    idx.append(names.index(ax))
                else:
                    idx = list(range(len(pos)-3, len(pos)))
                    break
            center = [pos[i] for i in idx]
            logger.info(f"Extracted center from ng_state['position'] using names: {center} (axes order: {names})")
        else:
            # If 4D or 5D, assume last 3 are z, y, x
            if len(pos) >= 3:
                center = pos[-3:]
            else:
                center = pos
            logger.info(f"Extracted center from ng_state['position'] using fallback: {center}")
    sources = get_layer_sources(ng_state, layer_name)

    if not sources:
        raise ValueError("No sources found for layer")

    # Debug transforms if requested
    if debug_transforms:
        analyze_transform_matrices(sources)

    # apply preference ordering
    if source_preference:
        def priority(s: SourceInfo):
            try:
                return source_preference.index(s.name)
            except ValueError:
                return len(source_preference)
        sources.sort(key=priority)

    logger.info("Found %d data source(s) for layer '%s':", len(sources), layer_name)
    for i, s in enumerate(sources):
        logger.info("  %d: %s -> %s", i + 1, s.name, s.url)

    mip_int = int(mip)
    accessible_sources = []
    for src in sources:
        if initialize_source(src, mip):
            accessible_sources.append(src)
            try:
                gmin, gmax = get_source_bounds_in_global_coords(src, mip_int)
                logger.info("  ✓ %s: global bounds %s to %s", src.name, gmin, gmax)
            except Exception:
                logger.info("  ✓ %s: initialized (no bounds)", src.name)
        else:
            logger.info("  ✗ %s: failed to initialize", src.name)

    if not accessible_sources:
        raise RuntimeError("No accessible data sources found")

    # Convert center from neuroglancer physical coords -> voxel coords using transform if present
    center_vox, matrix = ng_to_voxel_coord(center, ng_state, layer_name)

    if offset is None:
        offset = np.array([0, 0, 0], dtype=int)
    else:
        offset = np.asarray(offset, dtype=int)

    logger.info("Center (voxel, pre-offset): %s; using offset %s", center_vox, offset)

    # Convert to global coords (account for offset and MIP scaling)
    center_global = (np.asarray(center_vox) - np.asarray(offset)) // (2 ** mip_int)
    size_mip = [int(s // (2 ** mip_int)) for s in size]

    region_start = tuple(int(c - s // 2) for c, s in zip(center_global, size_mip))
    region_stop = tuple(int(c + s // 2) for c, s in zip(center_global, size_mip))
    region_shape = tuple(stop - start for start, stop in zip(region_start, region_stop))
    logger.info("Requested region: %s to %s (shape %s)", region_start, region_stop, region_shape)

    # find overlapping sources
    overlapping = find_sources_for_region(accessible_sources, region_start, region_stop, mip_int)
    if not overlapping:
        raise RuntimeError("No data sources overlap requested region")
    logger.info("Found %d overlapping sources", len(overlapping))

    # single-source fast path
    if len(overlapping) == 1 or not enable_cross_source:
        src = overlapping[0]
        translation = src.translation_voxel // (2 ** mip_int)
        local_start = tuple(int(region_start[i] - translation[i]) for i in range(3))
        local_stop = tuple(int(region_stop[i] - translation[i]) for i in range(3))
        # clip to source bounds
        smin, smax = src.bounds if src.bounds is not None else (local_start, local_stop)
        local_start = tuple(max(local_start[i], smin[i]) for i in range(3))
        local_stop = tuple(min(local_stop[i], smax[i]) for i in range(3))
        cutout = src.store[0, 0,
                           local_start[0]:local_stop[0],
                           local_start[1]:local_stop[1],
                           local_start[2]:local_stop[2]].read().result()
        cutout = np.asarray(cutout)
        merge_info = {
            "sources_used": [{"source_name": src.name, "source_idx": 0}],
            "merge_strategy": "single_source",
            "unfilled_voxels": 0,
            "total_voxels": int(np.prod(cutout.shape)),
            "coverage_ratio": 1.0
        }
    else:
        cutout, merge_info = merge_cutouts_from_sources(overlapping, region_start, region_stop, mip_int, merge_strategy)

    # Apply affine transform if present (matrix returned from ng_to_voxel_coord)
    if matrix is not None:
        logger.info("Applying affine transformation")
        if matrix.shape == (4, 4):
            mat3 = matrix[:3, :3]
            offset3 = matrix[:3, 3]
        else:
            mat3 = np.eye(3)
            offset3 = np.zeros(3)
        # compute center-based offset to avoid translation artifacts
        cut_shape = np.asarray(cutout.shape)
        center_in = (cut_shape - 1) / 2.0
        center_out = mat3.dot(center_in) + offset3
        offset_transform = offset3 + center_in - center_out
        cutout = affine_transform(cutout, mat3, offset=offset_transform, order=1)

    # Prepare output file paths
    src_names_list = [s["source_name"] for s in merge_info.get("sources_used", [])[:3]]
    source_names = "_".join(src_names_list)
    if len(merge_info.get("sources_used", [])) > 3:
        source_names += f"_and_{len(merge_info['sources_used']) - 3}_more"

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{source_names}_mip{mip}_{post_fix or now}"

    if out_path is None:
        out_path = base_name + ".tif"
    elif not out_path.endswith(".tif"):
        # if out_path is a directory, join; else append filename
        if os.path.isdir(out_path) or out_path.endswith(os.sep):
            out_path = os.path.join(out_path, base_name + ".tif")
        else:
            out_path = out_path + ".tif"

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    tif_path = None
    if save_tif:
        logger.info("Saving cutout to %s with shape %s", out_path, cutout.shape)
        # transpose to z,y,x -> x,y,z if desired; original code transposed to (2,1,0) before return
        tiff.imwrite(out_path, np.transpose(cutout, (2, 1, 0)))
        tif_path = os.path.abspath(out_path)

    # Write metadata
    meta_path = os.path.splitext(out_path)[0] + "_metadata.json"
    metadata = {
        "NG_link": ng_link,
        "image_tile": layer_name,
        "center": center_vox.tolist() if isinstance(center_vox, np.ndarray) else center_vox,
        "size": size,
        "mip": mip,
        "tif_path": tif_path,
        "metadata_path": os.path.abspath(meta_path),
        "merge_info": merge_info,
        "total_sources_found": len(sources),
        "accessible_sources": len(accessible_sources),
        "overlapping_sources": len(overlapping),
        "enable_cross_source": enable_cross_source,
        "merge_strategy": merge_strategy
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Cutout completed! Coverage: %.1f%%", 100.0 * merge_info.get("coverage_ratio", 0.0))
    logger.info("Used %d source(s) with %d unfilled voxels", len(merge_info.get("sources_used", [])), merge_info.get("unfilled_voxels", 0))

    # Return final array in the same orientation as original script (transpose to x,y,z)
    return cutout.transpose((2, 1, 0)), metadata


# -----------------------
# Backwards compat wrapper
# -----------------------
def download_cutout_from_zarr(ng_link, layer_name, center, size, out_path=None, post_fix=None, mip="0", offset=None, save_tif=True):
    return cutout_from_NG(
        ng_link=ng_link, layer_name=layer_name, center=center, size=size,
        out_path=out_path, post_fix=post_fix, mip=mip, offset=offset, save_tif=save_tif,
        enable_cross_source=False
    )


# -----------------------
# CLI entrypoint
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download a cutout that may span multiple data sources.")
    parser.add_argument("--ng_link", type=str, required=True, help="Neuroglancer state URL or '#!' fragment or JSON string")
    parser.add_argument("--layer", type=str, default=None, help="Layer name in the Neuroglancer state (default: first layer)")
    parser.add_argument("--center", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"), help="Center (z y x); default: position from the Neuroglancer state")
    parser.add_argument("--size", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"), help="Size (z y x) of the requested cutout")
    parser.add_argument("--out", type=str, default=None, help="Output path or filename (optional)")
    parser.add_argument("--mip", type=str, default="0", help="MIP level (default 0)")
    parser.add_argument("--no-save-tif", dest="save_tif", action="store_false", help="Do not save TIFF")
    parser.add_argument("--merge-strategy", type=str, default="first_valid", choices=["first_valid", "average", "max", "min"])
    args = parser.parse_args()

    cutout, meta = cutout_from_NG(
        ng_link=args.ng_link,
        layer_name=args.layer,
        center=tuple(args.center) if args.center else None,
        size=tuple(args.size),
        out_path=args.out,
        mip=args.mip,
        save_tif=args.save_tif,
        merge_strategy=args.merge_strategy
    )
    logger.info("Done. Metadata saved to %s", meta["metadata_path"])


if __name__ == "__main__":
    main()
