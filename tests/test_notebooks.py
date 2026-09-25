import json
from pathlib import Path


def test_eda_notebook_is_valid_json_with_expected_sections():
    nb_path = Path("notebooks/01_eda.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text())
    assert nb["nbformat"] == 4
    source_text = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    for expected in ["read_parquet", "visit", "groupby", "treatment"]:
        assert expected in source_text
