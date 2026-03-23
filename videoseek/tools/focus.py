import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.utils import call_llm_api, convert_to_free_form_text_representation
from config import general_config


focus_tool = {
    "type": "function",
    "function": {
        "name": "focus",
        "description": "To verify fine visual details, dense inspection of a short clip (start_time - end_time,≤ {focus_num_frames}s, at 1 FPS).".format(focus_num_frames=general_config["frame_sampling_factor"] * general_config["focus_base"]),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to focus the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to focus.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to focus.",
                },
            },
            "required": ["query", "start_time", "end_time"],
            "additionalProperties": False,
        },
    },
}


def execute_focus(config: dict, parameters: dict) -> str:
    """
    Execute the focus tool.
    """
    query = parameters["query"]
    start_time = parameters["start_time"]
    end_time = parameters["end_time"]
    max_num_frames = general_config["frame_sampling_factor"] * general_config["focus_base"]
    vr = parameters["vr"]
    subtitles = parameters["subtitles"]
    subtitles_str = convert_to_free_form_text_representation(
        [
            subtitle
            for subtitle in subtitles
            if float(subtitle["start_time"]) <= float(end_time)
            or float(subtitle["end_time"]) >= float(start_time)
        ],
        content_type="subtitle",
    )

    # Frame Sampling
    start_frame = int(start_time * vr.get_avg_fps())
    end_frame = min(int(end_time * vr.get_avg_fps()), len(vr) - 1)
    num_frames = min(int(end_frame - start_frame), max_num_frames)
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    cur_timestamps = np.array(
        [round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    frames = vr.get_batch(frame_indices).asnumpy()

    content = [{"type": "text", "text": f"Video clip ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        content.append({"type": "text", "text": f"{timestamp:.1f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    content.append(
        {
            "type": "text",
            "text": (
                "Video Subtitles:\n"
                f"{subtitles_str}\n\n"
                f"Question:\n{query}\n\n"
                "Please answer the question based on the given video clip. "
                "If the clip is not related to the question, please return 'No relevant content found.'"
            )
        }
    )

    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config["api_version"],
        max_tokens=config["max_tokens"],
        reasoning_effort=config["reasoning_effort"],
        seed=config["seed"],
        temperature=config["temperature"],
    )
    
    return response.choices[0].message.content