import base64
from io import BytesIO

import numpy as np
from PIL import Image

from videoseek.utils import call_llm_api, convert_to_free_form_text_representation
from config import general_config


skim_tool = {
    "type": "function",
    "function": {
        "name": "skim",
        "description": "To localize moments related to the query, quickly scan of a long segment (> {skim_num_frames}s) by sampling {skim_num_frames} frames from the video segment (start_time - end_time).".format(skim_num_frames=general_config["frame_sampling_factor"] * general_config["skim_base"]),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to skim the video. The query should be a concise question that can be answered by the video.",
                },
                "start_time": {
                    "type": "number",
                    "description": "The start time of the video to skim.",
                },
                "end_time": {
                    "type": "number",
                    "description": "The end time of the video to skim.",
                },
            },
            "required": ["query", "start_time", "end_time"],
            "additionalProperties": False,
        },
    },
}


def execute_skim(config: dict, parameters: dict) -> str:
    """
    Execute the skim tool.
    """
    query = parameters["query"]
    start_time = parameters["start_time"]
    end_time = parameters["end_time"]
    num_frames = general_config["frame_sampling_factor"] * general_config["skim_base"]
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
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).astype(int)
    cur_timestamps = np.array(
        [round(frame_indice / vr.get_avg_fps(), 1) for frame_indice in frame_indices],
        dtype=np.float32,
    )
    frames = vr.get_batch(frame_indices).asnumpy()

    # resize the shorter side to 256
    _, height, width, _ = frames.shape
    short_side = min(height, width)
    scale = 256 / short_side
    target_height = max(1, int(round(height * scale)))
    target_width = max(1, int(round(width * scale)))
    resized_frames = [
        np.array(
            Image.fromarray(frame).resize(
                (target_width, target_height),
                Image.BICUBIC,
            )
        )
        for frame in frames
    ]
    frames = np.stack(resized_frames, axis=0).reshape(
        num_frames, target_height, target_width, 3
    )

    content = [{"type": "text", "text": f"Video segment ({start_time:.1f}s - {end_time:.1f}s):\n"}]
    for frame, timestamp in zip(frames, cur_timestamps):
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="jpeg")
        byte_data = output_buffer.getvalue()
        base64_image = base64.b64encode(byte_data).decode("utf-8")
        content.append({
            "type": "text",
            "text": f"{timestamp:.1f}s",
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        })
    content.append({
        "type": "text",
        "text": f"Video Subtitles:\n{subtitles_str}\n\nQuestion:\n{query}\n\nPlease describe the content of the viewed video frames in detail with their timestamps (each frame with ~25 words). If query related content is found, please highlight the timestamps of the video frames that are relevant to the question and explain why (each timestamp with additional ~50 words). Do not answer the question directly.",
    })

    response = call_llm_api(
        messages=[{"role": "user", "content": content}],
        model_name=config["model_name"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        api_version=config["api_version"],
        max_tokens=config["max_tokens"],
        reasoning_effort="low",
        seed=config["seed"],
        temperature=config["temperature"],
    )

    return response.choices[0].message.content
