"""Exhibit registry, in manuscript order. Each module defines NAME, KIND, TITLE and build(ex)."""
from importlib import import_module

ORDER = ("table2_primary_id", "fig1_rank_composition", "fig2_geometry_discovery", "tableB1_extended_id",
         "tableB2_ten_encoder_robustness", "figB3_frozen_transfer_rank", "tableA1_dataset_composition",
         "tableA2_encoders", "fig2_geometry_discovery_scatter")


def get(name: str):
    if name not in ORDER:
        raise KeyError(f"unknown exhibit {name!r}; known: {', '.join(ORDER)}")
    return import_module(f"{__name__}.{name}")
