"""mx_custom_fallback: 零侵入式自定义 ModelExpress 回退源注入。

当 modelexpress 的 P2P source 查找失败时，默认回退到本地磁盘 / HF Hub 加载。
本包通过 monkey-patch ``SglangAdapter.load_via_native``，将回退源替换为可配置的
自定义源（本地路径、HF repo、S3/GCS/Azure、HTTP、Mooncake TransferEngine 等）。

使用方式:
    # 1. 设置环境变量
    export MX_FALLBACK_SOURCE_URL=/data/models/my-model   # 或 s3://bucket/model 或 HF repo ID
    # 可选: export MX_FALLBACK_SOURCE_TYPE=auto  # auto|local|hf|model_streamer|http|mooncake|mooncake_store
    # 可选: export MX_FALLBACK_FORCE=1           # 跳过 P2P 直接用回退源

    # Mooncake TransferEngine 回退 (从 seed 实例 RDMA 拉取):
    # 配置使用 Mooncake 原生环境变量 (MOONCAKE_*) 和 sglang 原生 load_config 字段
    export MX_FALLBACK_SOURCE_TYPE=mooncake
    # seed URL 来自 sglang load_config: --remote-instance-weight-loader-seed-instance-ip + --service-port
    # 或通过 MOONCAKE_META_FILE 指定元数据文件:
    export MOONCAKE_META_FILE=/tmp/mx_meta.json
    export MOONCAKE_PROTOCOL=rdma    # Mooncake 原生环境变量
    export MOONCAKE_DEVICE=erdma_0   # Mooncake 原生环境变量

    # MooncakeDistributedStore 回退 (从 Mooncake Store 加载):
    export MX_FALLBACK_SOURCE_TYPE=mooncake_store
    export MX_FALLBACK_SOURCE_URL=mooncake_store://my-model  # model name in store
    export MOONCAKE_MASTER=10.0.0.1:50051    # Mooncake 原生环境变量
    export MOONCAKE_PROTOCOL=rdma             # Mooncake 原生环境变量
    export MOONCAKE_DEVICE=erdma_0            # Mooncake 原生环境变量

    # 2. 在 sglang 启动前 import 本包
    import mx_custom_fallback

    # 3. 正常启动 sglang (使用 modelexpress backend)
    # python -m sglang.launch_server --load-format remote_instance \
    #   --remote-instance-weight-loader-backend modelexpress ...

零修改: 不改 sglang 代码，不改 modelexpress 代码。
原理: 两条路径 (nixl / transfer_engine) 最终都经过 ``SglangAdapter.load_via_native()``，
      patch 该方法即可覆盖所有回退场景。

Mooncake 回退原理:
  模型已初始化（随机权重在 GPU 上），Mooncake TransferEngine 通过 RDMA
  直接从 seed 实例的显存拉取权重覆写本地 tensor，无需磁盘 I/O。
  最大化复用 sglang 原生函数:
    - get_remote_instance_transfer_engine_info_per_rank()  获取 seed 信息
    - register_memory_region()  注册本地模型显存
    - _post_load_weights()  权重后处理
    - get_local_ip_auto() + envs.MOONCAKE_*  TransferEngine 初始化
  MooncakeDistributedStore 路径复用 sglang MooncakeStoreConfig.load_from_env()。
  配置完全使用 Mooncake 原生环境变量 (MOONCAKE_PROTOCOL, MOONCAKE_DEVICE, MOONCAKE_MASTER 等)。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("mx_custom_fallback")

# 在 import 时读取配置
FALLBACK_SOURCE_URL: str = os.environ.get("MX_FALLBACK_SOURCE_URL", "")
FALLBACK_SOURCE_TYPE: str = os.environ.get("MX_FALLBACK_SOURCE_TYPE", "auto")
FALLBACK_FORCE: bool = os.environ.get("MX_FALLBACK_FORCE", "0").lower() in (
    "1", "true", "yes", "on",
)

_installed = False


def install() -> bool:
    """执行 monkey-patch。返回 True 表示已安装，False 表示跳过（未配置源）。"""
    global _installed
    if _installed:
        return True

    if not FALLBACK_SOURCE_URL:
        logger.info(
            "[mx_custom_fallback] MX_FALLBACK_SOURCE_URL not set, "
            "skipping install (native fallback unchanged)"
        )
        return False

    from mx_custom_fallback.patcher import install_patches

    install_patches(
        source_url=FALLBACK_SOURCE_URL,
        source_type=FALLBACK_SOURCE_TYPE,
        force=FALLBACK_FORCE,
    )
    _installed = True
    logger.info(
        "[mx_custom_fallback] Installed: source_url=%s source_type=%s force=%s",
        FALLBACK_SOURCE_URL, FALLBACK_SOURCE_TYPE, FALLBACK_FORCE,
    )
    return True


# 模块被 import 时自动安装
install()
