import json
from pathlib import Path
from importlib.metadata import version


from noob import AssetSpecification, InputSpecification, NodeSpecification, TubeSpecification

REPO_ROOT = Path(__file__).parents[1]
SCHEMA_DIR = REPO_ROOT / "schema"
TARGETS = {
    SCHEMA_DIR / "tube.schema.json": TubeSpecification,
    SCHEMA_DIR / "node.schema.json": NodeSpecification,
    SCHEMA_DIR / "asset.schema.json": AssetSpecification,
    SCHEMA_DIR / "input.schema.json": InputSpecification,
}

VERSION_TAG = ".".join(version("noob").split(".")[:3])
BASE_URL = f"https://github.com/miniscope/noob/tree/v{VERSION_TAG}/schema/"


def render_schema() -> None:
    SCHEMA_DIR.mkdir(exist_ok=True)
    for path, model in TARGETS.items():
        with open(path, "w") as f:
            schema = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": BASE_URL + path.name,
                **model.model_json_schema(by_alias=True),
            }
            if model is TubeSpecification:
                # replace the default version, which has git commit qualifiers and whatnot,
                # with the simple tag version
                schema["properties"]["noob_version"]["default"] = VERSION_TAG
            json.dump(schema, f, indent=2)


if __name__ == "__main__":
    render_schema()
