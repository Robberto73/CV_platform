from __future__ import annotations

import traceback
from typing import Callable, Optional

from pipeline.graph import build_graph
from pipeline.state import PipelineState
from pipeline.utils.logging_utils import ProgressReporter
from pipeline.save_results import make_output_dir


def run_pipeline(
    state: PipelineState,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> PipelineState:
    graph = build_graph()
    state["progress_cb"] = progress_cb
    pr = ProgressReporter(progress_cb)
    try:
        return graph.invoke(state)
    except Exception as e:
        state.setdefault("processing_log", [])
        pr.log(state, f"runner: exception: {type(e).__name__}: {e}")
        pr.log(state, traceback.format_exc())
        state["error"] = True
        state["error_code"] = "EXCEPTION"
        state["error_message"] = f"{type(e).__name__}: {e}"
        if not state.get("output_dir"):
            state["output_dir"] = make_output_dir("out", state.get("user_query", ""))
        # best-effort: попытаться сохранить результаты с ошибкой
        from pipeline.nodes import error_handler, save_results

        state = error_handler(state)
        state = save_results(state)
        return state

