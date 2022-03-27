"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from task1.pipelines.data_science import create_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    
    ds_pipeline = create_pipeline()

    return {
        "ds": ds_pipeline,
        "__default__": pipeline([])
    }
