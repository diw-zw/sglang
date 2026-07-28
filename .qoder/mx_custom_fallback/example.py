#!/usr/bin/env python3
"""mx_custom_fallback 使用示例。

三种激活方式 (任选其一):

=== 方式 1: 环境变量 + 手动 import (最简单) ===

    export MX_FALLBACK_SOURCE_URL=/data/models/Qwen2.5-7B-Instruct
    # 或:
    export MX_FALLBACK_SOURCE_URL=s3://my-bucket/models/Qwen2.5-7B
    export MX_FALLBACK_SOURCE_TYPE=model_streamer

    python -c "import mx_custom_fallback; import sglang; ..."

=== 方式 2: pip install 自动导入 (零代码修改) ===

    cd /path/to/sglang/.qoder/mx_custom_fallback
    pip install -e .
    export MX_FALLBACK_SOURCE_URL=/data/models/Qwen2.5-7B
    # 之后任何 Python 进程启动时自动激活

=== 方式 3: PYTHONPATH + 启动脚本注入 ===

    export PYTHONPATH=/path/to/sglang/.qoder:$PYTHONPATH
    export MX_FALLBACK_SOURCE_URL=/data/models/Qwen2.5-7B

    # 在 sglang 启动脚本开头加一行:
    # import mx_custom_fallback

=== 配置环境变量说明 ===

    MX_FALLBACK_SOURCE_URL    必填。回退源 URL/路径
                              - 本地路径: /data/models/my-model
                              - HF repo:  Qwen/Qwen2.5-7B-Instruct 或 hf:Qwen/Qwen2.5-7B
                              - S3:       s3://bucket/path/to/model
                              - GCS:      gs://bucket/path/to/model
                              - HTTP:     https://example.com/models/my-model/

    MX_FALLBACK_SOURCE_TYPE   可选。源类型 (默认 auto 自动检测)
                              - auto:           根据 URL scheme 自动判断
                              - local:          本地目录
                              - hf:             HuggingFace repo ID
                              - model_streamer: S3/GCS/Azure (需 runai_model_streamer)
                              - http:           HTTP(S) 下载
                              - mooncake:       Mooncake TransferEngine RDMA 拉取
                              - mooncake_store: MooncakeDistributedStore 加载

    === Mooncake TransferEngine 回退 (使用 Mooncake 原生配置) ===

    export MX_FALLBACK_SOURCE_TYPE=mooncake

    # seed URL 来自 sglang 原生 load_config:
    #   --remote-instance-weight-loader-seed-instance-ip <ip>
    #   --remote-instance-weight-loader-seed-instance-service-port <port>
    # 或通过 MOONCAKE_META_FILE 指定元数据文件:
    export MOONCAKE_META_FILE=/tmp/mx_meta.json

    # TransferEngine 配置使用 Mooncake 原生环境变量:
    export MOONCAKE_PROTOCOL=rdma     # 传输协议
    export MOONCAKE_DEVICE=erdma_0    # RDMA 设备名

    === MooncakeDistributedStore 回退 (使用 Mooncake 原生配置) ===

    export MX_FALLBACK_SOURCE_TYPE=mooncake_store
    export MX_FALLBACK_SOURCE_URL=mooncake_store://my-model  # store 中的 model name
    # 或: export MOONCAKE_MODEL_NAME=my-model

    # Store 配置使用 Mooncake 原生环境变量:
    export MOONCAKE_MASTER=10.0.0.1:50051      # Master server (必需)
    # 或: export MOONCAKE_CLIENT=10.0.0.1:50052  # Client server (替代 MASTER)
    export MOONCAKE_PROTOCOL=rdma               # 传输协议
    export MOONCAKE_DEVICE=erdma_0              # RDMA 设备名
    export MOONCAKE_LOCAL_HOSTNAME=node-01      # 本地主机名
    export MOONCAKE_GLOBAL_SEGMENT_SIZE=4gb     # 全局段大小

    MX_FALLBACK_FORCE         可选。设为 1 跳过 P2P 发现，直接用回退源
                              (适用于已知无 P2P source 的场景)

    MX_FALLBACK_NO_AUTOIMPORT 可选。设为 1 禁用 .pth 自动导入
                              (仅方式 2 受影响)
"""

import os
import sys

# === 配置回退源 ===
# 在这里设置，或通过环境变量在启动前设置

# 方式 A: 本地/HF/S3/HTTP 回退
os.environ.setdefault("MX_FALLBACK_SOURCE_URL", "/data/models/example-model")
os.environ.setdefault("MX_FALLBACK_SOURCE_TYPE", "auto")

# 方式 B: Mooncake TransferEngine 回退 (取消注释即可)
# 使用 Mooncake 原生环境变量配置 (MOONCAKE_*)
# os.environ.setdefault("MX_FALLBACK_SOURCE_TYPE", "mooncake")
# os.environ.setdefault("MOONCAKE_PROTOCOL", "rdma")
# os.environ.setdefault("MOONCAKE_DEVICE", "erdma_0")
# os.environ.setdefault("MOONCAKE_META_FILE", "/tmp/mx_meta.json")
# os.environ.setdefault("MX_FALLBACK_FORCE", "1")

# 方式 C: MooncakeDistributedStore 回退 (取消注释即可)
# 使用 Mooncake 原生环境变量配置 (MOONCAKE_*)
# os.environ.setdefault("MX_FALLBACK_SOURCE_TYPE", "mooncake_store")
# os.environ.setdefault("MX_FALLBACK_SOURCE_URL", "mooncake_store://my-model")
# os.environ.setdefault("MOONCAKE_MASTER", "10.0.0.1:50051")
# os.environ.setdefault("MOONCAKE_PROTOCOL", "rdma")
# os.environ.setdefault("MOONCAKE_DEVICE", "erdma_0")

# === 激活注入 ===
# 确保 .qoder 目录在 Python 路径中
_qoder_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _qoder_dir not in sys.path:
    sys.path.insert(0, _qoder_dir)

import mx_custom_fallback  # noqa: E402  import 即激活 monkey-patch

# 验证安装状态
from mx_custom_fallback.patcher import _original_load_via_native

print(f"[example] mx_custom_fallback installed: {_original_load_via_native is not None}")
print(f"[example] fallback source: {mx_custom_fallback.FALLBACK_SOURCE_URL}")
print(f"[example] fallback type:   {mx_custom_fallback.FALLBACK_SOURCE_TYPE}")
print(f"[example] force mode:      {mx_custom_fallback.FALLBACK_FORCE}")

# === 之后正常启动 sglang ===
# from sglang import Engine
# engine = Engine(
#     model_path="my-model",
#     load_format="remote_instance",
#     remote_instance_weight_loader_backend="modelexpress",
#     modelexpress_config='{"url": "localhost:8001", "transport": "nixl"}',
#     ...
# )
