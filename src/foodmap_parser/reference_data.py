"""Load versioned FoodMap reference tables without changing the v0.1 baseline."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EXTENSION_FILENAMES = {
    "food_items.csv": "food_items_extension.csv",
    "food_aliases.csv": "food_aliases_extension.csv",
    "component_vocab.csv": "component_vocab_extension.csv",
    "component_aliases.csv": "component_aliases_extension.csv",
    "food_component_annotations.csv": "food_component_annotations_extension.csv",
}


def load_reference_csv(filename):
    """Load the stable baseline plus any reviewed v0.3 extension rows."""
    paths = [DATA_DIR / "v0.1" / filename]
    extension_name = EXTENSION_FILENAMES.get(filename)
    if extension_name:
        extension_path = DATA_DIR / "v0.3" / extension_name
        if extension_path.exists():
            paths.append(extension_path)

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file, skipinitialspace=True))
    return rows


def source_id_for_annotation(annotation):
    """Preserve whether a component link came from the baseline or extension."""
    if annotation.get("version", "").startswith("v0.3"):
        return "SRC_FOODMAP_CURATED_V03_EXPANSION"
    return "SRC_FOODMAP_CURATED_V01"
