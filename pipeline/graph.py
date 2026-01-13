from __future__ import annotations

from langgraph.graph import END, StateGraph

from pipeline.nodes import (
    error_handler,
    generate_final_answer,
    parse_user_request,
    prepare_key_frames,
    run_llava_analysis,
    save_results,
    select_video_analysis_mode,
)
from pipeline.state import PipelineState


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("parse_user_request", parse_user_request)
    g.add_node("select_video_analysis_mode", select_video_analysis_mode)
    g.add_node("prepare_key_frames", prepare_key_frames)
    g.add_node("run_llava_analysis", run_llava_analysis)
    g.add_node("generate_final_answer", generate_final_answer)
    g.add_node("save_results", save_results)
    g.add_node("error_handler", error_handler)

    g.set_entry_point("parse_user_request")
    g.add_edge("parse_user_request", "select_video_analysis_mode")
    g.add_edge("select_video_analysis_mode", "prepare_key_frames")

    def _after_prepare(state: PipelineState) -> str:
        key_frames = state.get("key_frames") or {}
        any_frames = any(len(v) > 0 for v in key_frames.values())
        if not any_frames or not state.get("require_visual_analysis", True):
            return "generate_final_answer"
        if state.get("vision_llm", "llava_local") == "off":
            return "generate_final_answer"
        return "run_llava_analysis"

    g.add_conditional_edges(
        "prepare_key_frames",
        _after_prepare,
        {
            "run_llava_analysis": "run_llava_analysis",
            "generate_final_answer": "generate_final_answer",
        },
    )

    g.add_edge("run_llava_analysis", "generate_final_answer")
    g.add_edge("generate_final_answer", "save_results")
    g.add_edge("save_results", END)

    # Примечание: в этой базовой версии перехват исключений делается в runner'е,
    # чтобы любой узел мог уйти в error_handler.

    return g.compile()

