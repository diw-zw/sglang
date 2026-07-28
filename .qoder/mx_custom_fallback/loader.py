"""自定义源加载逻辑。

根据 ``MX_FALLBACK_SOURCE_TYPE`` 将权重从不同类型的源加载到已初始化的 sglang 模型中。

支持的源类型:
  - ``auto`` (默认): 根据 URL scheme 自动检测
  - ``local`` / ``hf``: 本地路径或 HuggingFace repo ID，通过 DefaultModelLoader 加载
  - ``model_streamer``: S3/GCS/Azure 对象存储，通过 RunaiModelStreamerLoader 加载
  - ``http``: HTTP(S) URL，先下载到临时目录再本地加载
  - ``mooncake``: 通过 Mooncake TransferEngine 从 seed 实例 RDMA 拉取权重
    配置使用 Mooncake 原生环境变量 (MOONCAKE_PROTOCOL, MOONCAKE_DEVICE 等)
    和 sglang 原生 load_config 字段，最大化复用 sglang 原生函数
  - ``mooncake_store``: 通过 MooncakeDistributedStore 从 Mooncake Store 加载权重
    配置使用 Mooncake 原生环境变量 (MOONCAKE_MASTER, MOONCAKE_PROTOCOL 等)

Mooncake 路径复用的 sglang 原生函数:
  - ``get_remote_instance_transfer_engine_info_per_rank()``: 获取 seed 信息
  - ``register_memory_region()``: 注册本地模型显存
  - ``_post_load_weights()``: 权重后处理
  - ``MooncakeStoreConfig.load_from_env()``: 从环境变量加载 Store 配置
  - ``get_local_ip_auto()``, ``envs.MOONCAKE_*``: TransferEngine 初始化

所有加载路径都复用 sglang 原生的 weight loading + postprocess 流程，
确保 quantization / process_weights_after_loading 等后处理正确执行。
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("mx_custom_fallback.loader")


def load_from_custom_source(
    *,
    adapter,
    result,
    source_url: str,
    source_type: str = "auto",
):
    """从自定义源加载权重到已初始化的模型。

    Args:
        adapter: SglangAdapter 实例（提供 load_config, model_config, target_device 等）。
        result: LoadResult 包裹器（含已初始化的 model）。
        source_url: 自定义源 URL/路径。
        source_type: 源类型 (auto|local|hf|model_streamer|http)。

    Returns:
        更新后的 LoadResult。
    """
    if source_type == "auto":
        source_type = _detect_source_type(source_url)

    logger.info(
        "[mx_custom_fallback] Loading from source: type=%s url=%s",
        source_type, source_url,
    )

    if source_type in ("local", "hf"):
        return _load_from_path_or_hf(adapter, result, source_url)
    elif source_type == "model_streamer":
        return _load_via_model_streamer(adapter, result, source_url)
    elif source_type == "http":
        return _load_from_http(adapter, result, source_url)
    elif source_type == "mooncake":
        return _load_via_mooncake(adapter, result, source_url)
    elif source_type == "mooncake_store":
        return _load_via_mooncake_store(adapter, result, source_url)
    else:
        logger.warning(
            "[mx_custom_fallback] Unknown source type %r, falling back to native",
            source_type,
        )
        from mx_custom_fallback.patcher import _original_load_via_native
        if _original_load_via_native is not None:
            return _original_load_via_native(adapter, result)
        raise RuntimeError(
            f"Unknown source type {source_type!r} and no original load_via_native available"
        )


# ---------------------------------------------------------------------------
# 源类型检测
# ---------------------------------------------------------------------------

_OBJECT_STORAGE_SCHEMES = ("s3://", "gs://", "az://", "azure://")


def _detect_source_type(url: str) -> str:
    """根据 URL scheme 自动检测源类型。"""
    if url.startswith(_OBJECT_STORAGE_SCHEMES):
        return "model_streamer"
    if url.startswith("mooncake_store://"):
        return "mooncake_store"
    if url.startswith("mooncake://"):
        return "mooncake"
    if url.startswith(("http://", "https://")):
        return "http"
    if url.startswith("hf:"):
        return "hf"
    # 检查是否是本地目录
    if os.path.isdir(url):
        return "local"
    # 默认当作 HF repo ID
    return "hf"


# ---------------------------------------------------------------------------
# 加载实现: 本地路径 / HuggingFace repo
# ---------------------------------------------------------------------------

def _load_from_path_or_hf(adapter, result, path: str):
    """从本地路径或 HuggingFace repo 加载权重。

    与原始 ``SglangAdapter.load_via_native`` 逻辑相同，但使用自定义的 model_path。
    """
    # 去掉 hf: 前缀
    if path.startswith("hf:"):
        path = path[3:]

    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.model_loader.loader import DefaultModelLoader

    disk_config = copy.copy(adapter.load_config)
    disk_config.load_format = LoadFormat.AUTO

    # 用自定义路径覆盖 model_config
    model_config = copy.copy(adapter.model_config)
    _safe_setattr(model_config, "model_path", path)

    disk_loader = DefaultModelLoader(disk_config)
    weights_iter = disk_loader._get_all_weights(model_config, result.model)
    DefaultModelLoader.load_weights_and_postprocess(
        result.model, weights_iter, adapter.target_device,
    )
    logger.info("[mx_custom_fallback] Weights loaded from: %s", path)
    return result


# ---------------------------------------------------------------------------
# 加载实现: 对象存储 (S3/GCS/Azure) via runai_model_streamer
# ---------------------------------------------------------------------------

def _load_via_model_streamer(adapter, result, uri: str):
    """通过 RunaiModelStreamerLoader 从对象存储流式加载权重。

    需要 ``runai_model_streamer`` 包已安装。
    复用 modelexpress SglangAdapter.build_model_streamer_weight_iter 的逻辑。
    """
    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.model_loader.loader import (
        DefaultModelLoader,
        RunaiModelStreamerLoader,
    )

    stream_config = copy.copy(adapter.load_config)
    _safe_setattr(stream_config, "load_format", LoadFormat.RUNAI_STREAMER)

    # 配置 extra_config
    extra_config = dict(getattr(stream_config, "model_loader_extra_config", None) or {})
    # 对象存储默认启用 distributed streaming
    extra_config.setdefault("distributed", True)
    _safe_setattr(stream_config, "model_loader_extra_config", extra_config)

    # 用自定义 URI 覆盖 model_weights
    stream_model_config = copy.copy(adapter.model_config)
    _safe_setattr(stream_model_config, "model_weights", uri)

    loader = RunaiModelStreamerLoader(stream_config)
    loader.target_device_str = str(adapter.target_device)
    weights_iter = loader._get_all_weights(stream_model_config, result.model)

    DefaultModelLoader.load_weights_and_postprocess(
        result.model, weights_iter, adapter.target_device,
    )
    logger.info("[mx_custom_fallback] Weights streamed from: %s", uri)
    return result


# ---------------------------------------------------------------------------
# 加载实现: HTTP(S) 下载
# ---------------------------------------------------------------------------

def _load_from_http(adapter, result, url: str):
    """从 HTTP(S) URL 下载模型文件到临时目录，然后本地加载。

    自动下载:
      - config.json, tokenizer 相关配置
      - model.safetensors.index.json (如果存在)
      - 所有 *.safetensors 文件 (根据 index)
      - 如果没有 index，尝试下载 model.safetensors

    对于大型模型，建议使用 ``model_streamer`` 类型直接流式加载。
    """
    import json
    import urllib.request

    base_url = url.rstrip("/")

    with tempfile.TemporaryDirectory(prefix="mx_fallback_") as tmpdir:
        logger.info(
            "[mx_custom_fallback] Downloading from %s to %s", base_url, tmpdir,
        )

        # 下载配置文件
        _http_download(base_url, "config.json", tmpdir)

        # 尝试下载 tokenizer 配置
        for tokenizer_file in [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ]:
            try:
                _http_download(base_url, tokenizer_file, tmpdir)
            except Exception:
                pass

        # 下载权重文件
        index_path = os.path.join(tmpdir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                index_data = json.load(f)
            weight_files = sorted(set(index_data.get("weight_map", {}).values()))
            logger.info(
                "[mx_custom_fallback] Found %d safetensors files in index",
                len(weight_files),
            )
            for wf in weight_files:
                _http_download(base_url, wf, tmpdir)
        else:
            # 没有 index，尝试直接下载单个 safetensors 文件
            try:
                _http_download(base_url, "model.safetensors", tmpdir)
            except Exception as e:
                logger.warning(
                    "[mx_custom_fallback] Could not download model.safetensors: %s", e,
                )

        # 从下载的临时目录本地加载
        return _load_from_path_or_hf(adapter, result, tmpdir)


def _http_download(base_url: str, filename: str, dest_dir: str) -> str:
    """下载单个文件到目标目录。返回本地文件路径。"""
    import urllib.request

    file_url = f"{base_url}/{filename}"
    dest_path = os.path.join(dest_dir, filename)

    # 创建子目录（如果有）
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    logger.debug("[mx_custom_fallback] Downloading: %s", file_url)
    t0 = time.perf_counter()
    urllib.request.urlretrieve(file_url, dest_path)
    elapsed = time.perf_counter() - t0

    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(
        "[mx_custom_fallback] Downloaded %s (%.1f MB, %.2fs)",
        filename, size_mb, elapsed,
    )
    return dest_path


# ---------------------------------------------------------------------------
# 加载实现: Mooncake TransferEngine (RDMA P2P) — 复用 sglang 原生函数
# ---------------------------------------------------------------------------

def _load_via_mooncake(adapter, result, source_url: str):
    """通过 Mooncake TransferEngine 从 seed 实例 RDMA 拉取权重。

    最大化复用 sglang 原生函数:
      - ``get_remote_instance_transfer_engine_info_per_rank()``: 获取 seed 信息
      - ``register_memory_region()``: 注册本地模型显存
      - ``_post_load_weights()``: 权重后处理 (via ``_post_load_weights_and_quant``)

    配置使用 Mooncake 原生环境变量和 sglang 原生 load_config 字段:

    **TransferEngine 初始化** (使用 Mooncake 原生环境变量):
      - ``MOONCAKE_PROTOCOL``: 传输协议 (默认: rdma)
      - ``MOONCAKE_DEVICE``: RDMA 设备名 (默认: 空字符串)

    **Seed 信息获取** (优先级递减):
      1. sglang 原生 load_config: ``remote_instance_weight_loader_seed_instance_ip``
         + ``remote_instance_weight_loader_seed_instance_service_port``
         → 复用 sglang ``get_remote_instance_transfer_engine_info_per_rank()``
      2. ``MOONCAKE_META_FILE`` 环境变量: 元数据 JSON 文件路径
      3. ``source_url``: ``mooncake://http://seed:port`` 或 meta 文件路径

    **TP rank**: 使用 ``load_config.tp_rank`` (sglang 原生)
    """
    model = result.model
    if model is None:
        raise RuntimeError("[mx_custom_fallback] Mooncake loading requires result.model")

    load_config = adapter.load_config
    tp_rank = getattr(load_config, "tp_rank", 0) or 0

    # --- 获取或创建 TransferEngine ---
    transfer_engine = getattr(
        load_config, "remote_instance_weight_loader_transfer_engine", None
    )
    if transfer_engine is not None:
        logger.info("[mx_custom_fallback] Reusing existing TransferEngine from load_config")
    else:
        transfer_engine = _init_mooncake_transfer_engine()

    # --- 解析 seed 来源 ---
    meta_file = os.environ.get("MOONCAKE_META_FILE", "")
    seed_url = ""

    if source_url.startswith("mooncake://"):
        parsed = source_url[len("mooncake://"):]
        if os.path.isfile(parsed):
            meta_file = meta_file or parsed
        else:
            seed_url = parsed
    elif os.path.isfile(source_url):
        meta_file = meta_file or source_url
    elif source_url:
        seed_url = source_url

    # 从 sglang load_config 获取 seed URL (原生方式)
    if not seed_url and not meta_file:
        seed_ip = getattr(load_config, "remote_instance_weight_loader_seed_instance_ip", None)
        seed_port = getattr(load_config, "remote_instance_weight_loader_seed_instance_service_port", None)
        if seed_ip and seed_port:
            seed_url = f"http://{seed_ip}:{seed_port}"

    # --- 获取 seed session_id 和 weight_info ---
    if meta_file:
        session_id, seed_weight_info = _get_seed_info_from_meta_file(meta_file)
    elif seed_url:
        # 复用 sglang 原生函数
        from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
            get_remote_instance_transfer_engine_info_per_rank,
        )
        logger.info(
            "[mx_custom_fallback] Fetching seed info via sglang native: %s?rank=%d",
            seed_url, tp_rank,
        )
        session_id, seed_weight_info = get_remote_instance_transfer_engine_info_per_rank(
            seed_url, tp_rank
        )
    else:
        raise RuntimeError(
            "[mx_custom_fallback] Mooncake fallback requires one of:\n"
            "  1. load_config.remote_instance_weight_loader_seed_instance_ip + service_port\n"
            "  2. MOONCAKE_META_FILE environment variable\n"
            "  3. source_url (mooncake://http://seed:port or path to meta.json)"
        )

    if not session_id or not seed_weight_info:
        raise RuntimeError("[mx_custom_fallback] Failed to get seed TransferEngine info")

    logger.info(
        "[mx_custom_fallback] Mooncake seed: session_id=%s, tensors=%d, tp_rank=%d",
        session_id, len(seed_weight_info), tp_rank,
    )

    # --- 注册本地模型显存 (复用 sglang 原生) ---
    from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
        register_memory_region,
    )
    register_memory_region(model, transfer_engine)
    logger.info("[mx_custom_fallback] Registered local model memory with TransferEngine")

    # --- 构建 transfer 列表并拉取 ---
    seed_ptr_list = []
    client_ptr_list = []
    client_len_list = []

    for name, tensor in model.named_parameters():
        weight_info = seed_weight_info.get(name)
        if weight_info is None:
            logger.warning("[mx_custom_fallback] Tensor %r not in seed, skipping", name)
            continue

        # 兼容两种 weight_info 格式:
        #   sglang bootstrap: (data_ptr, numel, element_size)
        #   mx_meta.json:     {"addr":..., "size":...}
        if isinstance(weight_info, (list, tuple)):
            seed_ptr, seed_numel, seed_element_size = weight_info
            seed_size = seed_numel * seed_element_size
        elif isinstance(weight_info, dict):
            seed_ptr = weight_info["addr"]
            seed_size = weight_info["size"]
        else:
            logger.warning("[mx_custom_fallback] Unexpected weight_info type for %r", name)
            continue

        client_ptr = tensor.data_ptr()
        client_len = tensor.numel() * tensor.element_size()

        if seed_size != client_len:
            raise RuntimeError(
                f"[mx_custom_fallback] Size mismatch for {name}: "
                f"seed={seed_size} bytes, local={client_len} bytes"
            )

        seed_ptr_list.append(seed_ptr)
        client_ptr_list.append(client_ptr)
        client_len_list.append(client_len)

    total_bytes = sum(client_len_list)
    logger.info(
        "[mx_custom_fallback] Mooncake transfer: %d tensors, %.2f GB",
        len(seed_ptr_list), total_bytes / 1e9,
    )

    t0 = time.perf_counter()
    ret = transfer_engine.batch_transfer_sync_read(
        session_id, client_ptr_list, seed_ptr_list, client_len_list,
    )
    elapsed = time.perf_counter() - t0

    if ret < 0:
        raise RuntimeError(
            f"[mx_custom_fallback] batch_transfer_sync_read failed: ret={ret}"
        )

    bandwidth_gbps = (total_bytes * 8) / (elapsed * 1e9) if elapsed > 0 else 0
    logger.info(
        "[mx_custom_fallback] Mooncake transfer complete: %.2f GB, %.2fs, %.1f Gbps",
        total_bytes / 1e9, elapsed, bandwidth_gbps,
    )

    # --- 后处理 (复用 sglang 原生 + 量化后处理) ---
    _post_load_weights_and_quant(model, adapter.target_device)
    logger.info("[mx_custom_fallback] Mooncake weight loading complete")
    return result


def _init_mooncake_transfer_engine():
    """使用 Mooncake 原生环境变量初始化 TransferEngine。

    复用 sglang ``get_local_ip_auto()`` 和 ``envs.MOONCAKE_*`` 配置，
    逻辑与 ``ModelRunner.remote_instance_init_transfer_engine()`` 一致。
    """
    from mooncake.engine import TransferEngine

    # 复用 sglang 原生工具
    from sglang.srt.environ import envs
    from sglang.srt.utils import get_local_ip_auto

    local_ip = get_local_ip_auto()
    protocol = envs.MOONCAKE_PROTOCOL.get()
    device = envs.MOONCAKE_DEVICE.get()

    te = TransferEngine()
    ret = te.initialize(local_ip, "P2PHANDSHAKE", protocol, device)
    if ret != 0:
        raise RuntimeError(
            f"[mx_custom_fallback] TransferEngine initialize failed: ret={ret}, "
            f"local_ip={local_ip}, device={device}, protocol={protocol}"
        )

    logger.info(
        "[mx_custom_fallback] TransferEngine initialized: local=%s:%s, device=%s, protocol=%s",
        local_ip, te.get_rpc_port(), device, protocol,
    )
    return te


def _get_seed_info_from_meta_file(meta_path: str):
    """从 mx_query.py 产出的 JSON 元数据文件读取 session_id 和 weight_info。

    JSON 格式 (与 mx_pull.py 兼容):
      {
        "session_id": "ip:port",
        "tensors": [
          {"name": "...", "addr": 123, "size": 456, "device_id": 0, "dtype": "..."},
          ...
        ]
      }
    """
    import json as _json

    logger.info("[mx_custom_fallback] Reading seed metadata from: %s", meta_path)
    with open(meta_path) as f:
        meta = _json.load(f)

    session_id = meta["session_id"]
    tensors = meta.get("tensors", [])

    weight_info = {}
    for t in tensors:
        weight_info[t["name"]] = {"addr": t["addr"], "size": t["size"]}

    return session_id, weight_info


# ---------------------------------------------------------------------------
# 共享: 权重后处理 (复用 sglang 原生)
# ---------------------------------------------------------------------------

def _post_load_weights_and_quant(model, target_device):
    """权重后处理: 复用 sglang ``_post_load_weights`` + 量化后处理。

    与 sglang 原生 TransferEngine 路径相比，额外执行 ``process_weights_after_loading``
    (与 modelexpress SglangAdapter.after_rdma_receive 一致)。
    """
    from sglang.srt.model_loader.loader import _post_load_weights, device_loading_context

    _post_load_weights(model)

    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)


# ---------------------------------------------------------------------------
# 加载实现: MooncakeDistributedStore — 完全复用 sglang 原生链路
# ---------------------------------------------------------------------------

def _load_via_mooncake_store(adapter, result, source_url: str):
    """通过 MooncakeDistributedStore 加载权重。

    完全复用 sglang 原生链路，零自定义加载逻辑:
      - ``MooncakeStoreConnector`` (via ``create_remote_connector``):
        使用 ``MooncakeStoreConfig.load_from_env()`` 读取 ``MOONCAKE_*`` 环境变量，
        初始化 MooncakeDistributedStore，处理 standalone_storage / contiguity / dedup
      - ``RemoteModelLoader._load_model_from_remote_kv()`` (loader.py:2483):
        batch_get_into (batch=256) + ``process_weights_after_loading`` + ``_post_load_weights``

    与 sglang ``--load-format remote`` + ``mooncake:///model_name`` 路径完全一致。

    配置完全使用 Mooncake 原生环境变量 (通过 ``MooncakeStoreConfig.load_from_env()``):
      - ``MOONCAKE_MASTER`` / ``MOONCAKE_CLIENT``: server 地址 (必需)
      - ``MOONCAKE_PROTOCOL``, ``MOONCAKE_DEVICE``, ``MOONCAKE_LOCAL_HOSTNAME`` 等

    模型名来源 (优先级递减):
      1. ``source_url``: ``mooncake_store://model_name``
      2. ``MOONCAKE_MODEL_NAME`` 环境变量
      3. ``load_config.model_path`` 的 basename
    """
    # --- 解析 model_name ---
    model_name = ""
    if source_url.startswith("mooncake_store://"):
        model_name = source_url[len("mooncake_store://"):]
    if not model_name:
        model_name = os.environ.get("MOONCAKE_MODEL_NAME", "")
    if not model_name:
        model_path = getattr(adapter.load_config, "model_path", "") or ""
        model_name = os.path.basename(model_path) if model_path else ""
    if not model_name:
        raise RuntimeError(
            "[mx_custom_fallback] MooncakeStore requires model_name. "
            "Set source_url=mooncake_store://model_name or MOONCAKE_MODEL_NAME env var."
        )

    logger.info("[mx_custom_fallback] MooncakeStore: model=%s", model_name)

    # 构造 mooncake:/// URL — MooncakeStoreConnector 的 URL 格式
    mooncake_url = f"mooncake:///{model_name}"

    # 完全复用 sglang 原生 connector + loader
    from sglang.srt.connector import create_remote_connector
    from sglang.srt.model_loader.loader import RemoteModelLoader

    with create_remote_connector(mooncake_url, device=str(adapter.target_device)) as client:
        # _load_model_from_remote_kv 内部完成:
        #   1. process_weights_after_loading (量化后处理)
        #   2. ShardedStateLoader._filter_subtensors (子张量处理)
        #   3. client.batch_get_into (zero-copy RDMA, batch=256)
        #   4. _post_load_weights (模型后处理)
        RemoteModelLoader._load_model_from_remote_kv(
            None,  # self — batch_get_into 路径不使用 self
            result.model,
            adapter.model_config,
            client,
        )

    logger.info("[mx_custom_fallback] MooncakeStore weight loading complete")
    return result


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _safe_setattr(obj: Any, name: str, value: Any) -> None:
    """安全设置属性，处理 frozen dataclass 等情况。"""
    try:
        setattr(obj, name, value)
    except AttributeError:
        object.__setattr__(obj, name, value)
