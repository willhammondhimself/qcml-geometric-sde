"""Consistency guards for the fusion channel taxonomy.

These lock the invariants the fusion layer relies on so the taxonomy cannot
silently drift from the detector registry:

  * ACTIVE_CHANNELS is exactly the union of OBSERVABLE_FAMILIES.
  * DEAD_CHANNELS never overlaps the active set.
  * HierarchicalFusion resolves every family when given canonical display names,
    and *warns* (rather than silently dropping) when a name is missing.

Note: callers must key ``channel_names`` by the canonical display name (the
OBSERVABLE_FAMILIES / HPO_CONFIGS key), NOT ``detector.name`` — some detectors
carry variant suffixes in ``.name`` (e.g. "Sectional Curvature Sign (0,1)",
"Effective State Dimension"). That mismatch is exactly what the warning below
surfaces.
"""

import logging

from qcml_geometry.fusion import (
    ACTIVE_CHANNELS,
    DEAD_CHANNELS,
    OBSERVABLE_FAMILIES,
    HierarchicalFusionDetector,
)


def _all_family_channels():
    return [ch for chans in OBSERVABLE_FAMILIES.values() for ch in chans]


def test_active_channels_equals_union_of_families():
    union = set(_all_family_channels())
    assert set(ACTIVE_CHANNELS) == union


def test_family_channel_names_are_unique():
    names = _all_family_channels()
    assert len(names) == len(set(names)), "a channel appears in more than one family"


def test_dead_channels_disjoint_from_active():
    assert set(DEAD_CHANNELS).isdisjoint(set(ACTIVE_CHANNELS))


def test_resolver_maps_every_family_for_canonical_names():
    names = _all_family_channels()
    det = HierarchicalFusionDetector(channel_names=names)
    resolved = det._resolve_families(len(names))

    # Every family resolves, and every channel maps to its own column.
    assert set(resolved) == set(OBSERVABLE_FAMILIES)
    flat = [idx for idxs in resolved.values() for idx in idxs]
    assert sorted(flat) == list(range(len(names)))


def test_resolver_warns_when_channel_missing(caplog):
    names = _all_family_channels()
    missing = names[0]
    truncated = names[1:]  # drop the first channel
    det = HierarchicalFusionDetector(channel_names=truncated)

    with caplog.at_level(logging.WARNING, logger="qcml_geometry.fusion"):
        resolved = det._resolve_families(len(truncated))

    # The dropped channel is reported, not silently swallowed.
    assert any(missing in rec.getMessage() for rec in caplog.records)
    # Resolution still succeeds for the remaining channels.
    flat = [idx for idxs in resolved.values() for idx in idxs]
    assert sorted(flat) == list(range(len(truncated)))
