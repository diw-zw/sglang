"""Monkey-patch 逻辑: 替换 modelexpress 的回退加载入口。

核心思路:
  modelexpress 的两条加载路径最终都经过 ``SglangAdapter.load_via_native()``:
    - nixl transport: LoadStrategyChain -> DefaultStrategy -> load_via_native()
    - transfer_engine transport: _load_model_via_transfer_engine() -> load_via_native()

  patch 该方法，在 P2P source 不存在时从可配置的自定义源加载权重。

可选: ``MX_FALLBACK_FORCE=1`` 时额外 patch ``RdmaStrategy.is_available``，
      跳过 P2P 尝试，直接使用回退源（适用于已知无 P2P source 的场景，省去发现时间）。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("mx_custom_fallback.patcher")

# 保存原始方法引用，用于卸载或回退
_original_load_via_native: Any = None
_original_rdma_is_available: Any = None


def install_patches(
    *,
    source_url: str,
    source_type: str,
    force: bool,
) -> None:
    """安装所有必要的 monkey-patch。

    Args:
        source_url: 自定义回退源 URL/路径。
        source_type: 源类型 (auto|local|hf|model_streamer|http)。
        force: True 则跳过 P2P 策略，直接走回退源。
    """
    _patch_load_via_native(source_url, source_type)

    if force:
        _patch_rdma_strategy_skip()


def _patch_load_via_native(source_url: str, source_type: str) -> None:
    """替换 SglangAdapter.load_via_native，将回退源重定向到自定义 URL。"""
    global _original_load_via_native

    from modelexpress.engines.sglang.adapter import SglangAdapter

    _original_load_via_native = SglangAdapter.load_via_native

    def patched_load_via_native(self, result):
        """当 P2P source 不存在时的自定义回退加载。"""
        from mx_custom_fallback.loader import load_from_custom_source

        logger.info(
            "[mx_custom_fallback] P2P source unavailable, "
            "loading from custom fallback: url=%s",
            source_url,
        )
        return load_from_custom_source(
            adapter=self,
            result=result,
            source_url=source_url,
            source_type=source_type,
        )

    SglangAdapter.load_via_native = patched_load_via_native
    logger.info("[mx_custom_fallback] Patched SglangAdapter.load_via_native")


def _patch_rdma_strategy_skip() -> None:
    """跳过 RdmaStrategy，使 P2P 发现不阻塞回退流程。

    当 ``MX_FALLBACK_FORCE=1`` 时调用。直接让 ``RdmaStrategy.is_available``
    返回 False，使 LoadStrategyChain 跳过 P2P 尝试。
    """
    global _original_rdma_is_available

    try:
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
    except ImportError:
        logger.warning(
            "[mx_custom_fallback] Could not import RdmaStrategy, "
            "force-skip not applied"
        )
        return

    _original_rdma_is_available = RdmaStrategy.is_available

    @staticmethod
    def _skipped_is_available(ctx):  # type: ignore[override]
        logger.info(
            "[mx_custom_fallback] MX_FALLBACK_FORCE=1, "
            "skipping RdmaStrategy (P2P) for worker %s",
            getattr(ctx, "global_rank", "?"),
        )
        return False

    # is_available 是实例方法，需要正确绑定
    RdmaStrategy.is_available = lambda self, ctx: _skipped_is_available(ctx)
    logger.info("[mx_custom_fallback] Patched RdmaStrategy.is_available -> False")


def uninstall() -> None:
    """恢复原始方法（用于测试或卸载）。"""
    global _original_load_via_native, _original_rdma_is_available

    if _original_load_via_native is not None:
        from modelexpress.engines.sglang.adapter import SglangAdapter

        SglangAdapter.load_via_native = _original_load_via_native
        _original_load_via_native = None
        logger.info("[mx_custom_fallback] Restored SglangAdapter.load_via_native")

    if _original_rdma_is_available is not None:
        try:
            from modelexpress.load_strategy.rdma_strategy import RdmaStrategy

            RdmaStrategy.is_available = _original_rdma_is_available
            _original_rdma_is_available = None
            logger.info("[mx_custom_fallback] Restored RdmaStrategy.is_available")
        except ImportError:
            pass
