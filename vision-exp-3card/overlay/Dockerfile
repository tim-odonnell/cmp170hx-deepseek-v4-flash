# Pure-Python overlay on the native build (same fork commit 4fe10bf + our patch 0001).
# /vllm is an editable install, so replacing the source file is the whole change.
FROM dsv4-vision:sm80-leinasi-4fe10bf
COPY vl_model.py /vllm/vllm/models/deepseek_v4/nvidia/vl_model.py
