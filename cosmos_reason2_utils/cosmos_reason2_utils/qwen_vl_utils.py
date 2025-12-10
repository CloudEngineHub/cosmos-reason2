import pydantic

"""qwen-vl-utils utilities.

These are maintained for backwards compatibility. New code should use the
official huggingface vision processor.
"""


class QwenVLUtilsConfig(pydantic.BaseModel):
    """Config for vision processing.

    Source:
    https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

    Attributes are sorted by priority. Higher priority attributes override lower
    priority attributes.
    """

    model_config = pydantic.ConfigDict(extra="allow", use_attribute_docstrings=True)

    resized_height: int | None = None
    """Max height of the image/video"""
    resized_width: int | None = None
    """Max width of the image/video"""

    min_pixels: int | None = None
    """Min frame pixels of the image/video"""
    max_pixels: int | None = None
    """Max frame pixels of the image/video"""
    total_pixels: int | None = None
    """Max total pixels of the image/video"""

    video_start: float | None = None
    """Start time of the video (seconds)"""
    video_end: float | None = None
    """End time of the video (seconds)"""

    nframes: int | None = None
    """Number of frames of the video"""

    fps: float | None = None
    """FPS of the video"""
    min_frames: int | None = None
    """Min frames of the video"""
    max_frames: int | None = None
    """Max frames of the video"""

def set_qwen_vl_utils_args(messages: list[dict], vision_kwargs: dict, /):
    """Set vision processor arguments.

    Args:
        messages: Chat messages.
        vision_kwargs: Keyword arguments for vision processor (see `QwenVLUtilsConfig`).
    """
    QwenVLUtilsConfig.model_validate(vision_kwargs)
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            content = [content]
        for msg in content:
            if isinstance(msg, dict) and msg.get("type", None) in [
                "image",
                "video",
            ]:
                msg |= vision_kwargs
