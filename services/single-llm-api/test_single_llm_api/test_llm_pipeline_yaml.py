import yaml
from pathlib import Path


def test_yaml_reads_ok():
    filename = Path(__file__).parents[1] / "src" / "single_llm_api" / "llm-pipeline.yaml"
    with open(filename) as f:
        pipeline_spec = yaml.safe_load(f)
    # sanity check some keys in the yaml
    assert set(pipeline_spec.keys()) == {"pipeline"}
    assert set(pipeline_spec["pipeline"].keys()) == {"pipeline-type", "kwargs", "model", "tokenizer"}
