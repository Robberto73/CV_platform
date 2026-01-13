from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ParseUserRequestResult(BaseModel):
    require_visual_analysis: bool = False
    required_models: list[
        Literal["yolov8-person", "yolov8-product", "reid-tracker", "zone-detector"]
    ] = Field(default_factory=list)
    json_required: bool = False


class FrameLLMAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    event_type: Optional[str] = None


class EventEntity(BaseModel):
    id: Optional[str] = None
    class_name: Optional[str] = Field(default=None, alias="class")
    bbox: Optional[list[float]] = None
    confidence: Optional[float] = None
    type: Optional[str] = None
    zone: Optional[str] = None


class Event(BaseModel):
    event_id: str
    video_path: str
    timestamp_sec: float
    frame_id: int
    event_type: str
    entities: list[EventEntity] = Field(default_factory=list)
    llava_analysis: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_frames: list[int] = Field(default_factory=list)

    def to_parquet_row(self) -> dict[str, Any]:
        return {
            "timestamp_sec": float(self.timestamp_sec),
            "video_path": self.video_path,
            "event_type": self.event_type,
            "entities_json": [e.model_dump(by_alias=True) for e in self.entities],
            "confidence": float(self.confidence),
            "event_id": self.event_id,
            "frame_id": int(self.frame_id),
            "llava_analysis": self.llava_analysis,
            "source_frames": self.source_frames,
        }


class FinalAnswerJSON(BaseModel):
    answer: str
    details: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    processing_time_sec: float = 0.0
    models_used: list[str] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    error: bool = True
    error_code: str = "UNKNOWN"
    message: str = "Unhandled error"

