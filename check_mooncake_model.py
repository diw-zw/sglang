#!/usr/bin/env python3
"""
check_mooncake_model.py - 检查 Mooncake 中的模型是否完整

用法:
  python3 check_mooncake_model.py --model qwen3-5-397b-a17b-fp8 --tp 4
"""

import os
import json
import argparse
from pathlib import Path

def get_env_config():
    """从环境变量获取 Mooncake 配置"""
    return {
        'local_hostname': os.getenv('MOONCAKE_LOCAL_HOSTNAME', 'volume-exporter'),
        'metadata_server': os.getenv('MOONCAKE_TE_META_DATA_SERVER', 'http://10.0.36.17:52856/metadata'),
        'protocol': os.getenv('MOONCAKE_PROTOCOL', 'rdma'),
        'device_name': os.getenv('MOONCAKE_DEVICE', ''),
        'master_server_address': os.getenv('MOONCAKE_MASTER', '10.0.36.17:52858'),
    }

def connect_to_mooncake():
    """连接到 Mooncake store"""
    from mooncake.store import MooncakeDistributedStore

    config = get_env_config()
    store = MooncakeDistributedStore()

    ret = store.setup(
        config['local_hostname'],
        config['metadata_server'],
        128*1024*1024*1024,  # 128GB
        64*1024*1024,  # 64MB
        config['protocol'],
        config['device_name'],
        config['master_server_address']
    )

    if ret != 0:
        raise RuntimeError(f"Failed to setup Mooncake store: {ret}")

    return store

def check_config_files(store, model_name):
    """检查配置文件"""
    config_files = [
        'config.json',
        'tokenizer_config.json',
        'tokenizer.json',
        'special_tokens_map.json',
        'preprocessor_config.json',
        'generation_config.json',
    ]

    print(f"\n📄 Config files for {model_name}:")
    found = 0
    for fname in config_files:
        key = f"{model_name}/files/{fname}"
        exists = store.is_exist(key) == 1
        if exists:
            size = store.get_size(key)
            print(f"  ✓ {fname}: {size:,} bytes")
            found += 1
        else:
            print(f"  ✗ {fname}: NOT FOUND")

    return found, len(config_files)

def check_safetensors_index(store, model_name):
    """检查 safetensors index 文件"""
    key = f"{model_name}/files/model.safetensors.index.json"
    exists = store.is_exist(key) == 1

    print(f"\n📋 Safetensors index:")
    if not exists:
        print(f"  ✗ model.safetensors.index.json: NOT FOUND")
        return None, 0

    data = store.get(key)
    index = json.loads(data.decode('utf-8'))
    size = store.get_size(key)
    print(f"  ✓ model.safetensors.index.json: {size:,} bytes")

    return index, len(index.get('weight_map', {}))

def check_tensor_keys(store, model_name, tp_size, weight_map):
    """检查 tensor keys"""
    print(f"\n🔍 Checking tensors (TP={tp_size})...")

    # 统计每个 rank 的 tensor 数量
    rank_counts = {i: 0 for i in range(tp_size)}
    rank_missing = {i: [] for i in range(tp_size)}

    # 获取所有 unique tensor names
    tensor_names = set(weight_map.values())
    total = len(tensor_names)

    print(f"  Total unique tensors: {total}")

    # 检查每个 rank
    for rank in range(tp_size):
        for tensor_name in tensor_names:
            key = f"{model_name}/keys/rank_{rank}/{tensor_name}"
            if store.is_exist(key) == 1:
                rank_counts[rank] += 1
            else:
                rank_missing[rank].append(tensor_name)

    # 打印结果
    print(f"\n📊 Tensor counts by rank:")
    all_complete = True
    for rank in range(tp_size):
        count = rank_counts[rank]
        missing = len(rank_missing[rank])
        status = "✓" if count == total else "✗"
        print(f"  {status} rank_{rank}: {count}/{total} tensors ({missing} missing)")

        if missing > 0 and missing <= 10:
            print(f"    Missing: {', '.join(rank_missing[rank][:10])}")
        elif missing > 10:
            print(f"    Missing: {', '.join(rank_missing[rank][:5])} ... and {missing-5} more")

        if count != total:
            all_complete = False

    return all_complete

def check_total_size(store, model_name, tp_size, weight_map):
    """估算总大小"""
    print(f"\n💾 Estimating total size...")

    total_bytes = 0
    sample_count = 0

    # 采样一些 tensor 来估算
    tensor_names = list(weight_map.values())
    sample_size = min(100, len(tensor_names))

    for i in range(0, len(tensor_names), len(tensor_names) // sample_size):
        tensor_name = tensor_names[i]
        # 检查 rank 0 的大小
        key = f"{model_name}/keys/rank_0/{tensor_name}"
        if store.is_exist(key) == 1:
            size = store.get_size(key)
            total_bytes += size * tp_size  # 假设每个 rank 大小相似
            sample_count += 1

    if sample_count > 0:
        avg_size = total_bytes / sample_count
        estimated_total = avg_size * len(tensor_names)

        # 加上 config 文件
        for fname in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                      'model.safetensors.index.json']:
            key = f"{model_name}/files/{fname}"
            if store.is_exist(key) == 1:
                estimated_total += store.get_size(key)

        print(f"  Sampled {sample_count} tensors")
        print(f"  Estimated total size: {estimated_total / (1024**3):.2f} GB")

    return estimated_total if sample_count > 0 else 0

def main():
    parser = argparse.ArgumentParser(description='Check Mooncake model completeness')
    parser.add_argument('--model', required=True, help='Model name (e.g., qwen3-5-397b-a17b-fp8)')
    parser.add_argument('--tp', type=int, required=True, help='Tensor parallel size')
    args = parser.parse_args()

    print(f"🔌 Connecting to Mooncake store...")
    store = connect_to_mooncake()

    try:
        # 1. 检查配置文件
        config_found, config_total = check_config_files(store, args.model)

        # 2. 检查 safetensors index
        index, tensor_count = check_safetensors_index(store, args.model)

        # 3. 检查 tensors
        if index:
            weight_map = index.get('weight_map', {})
            tensors_complete = check_tensor_keys(store, args.model, args.tp, weight_map)

            # 4. 估算大小
            estimated_size = check_total_size(store, args.model, args.tp, weight_map)

        # 总结
        print(f"\n{'='*60}")
        print(f"📊 Summary:")
        print(f"  Config files: {config_found}/{config_total}")
        print(f"  Safetensors index: {'✓' if index else '✗'}")
        if index:
            print(f"  Tensors: {'✓ Complete' if tensors_complete else '✗ Incomplete'}")
            print(f"  Estimated size: {estimated_size / (1024**3):.2f} GB")

        if config_found == config_total and index and tensors_complete:
            print(f"\n✅ Model {args.model} is COMPLETE!")
        else:
            print(f"\n❌ Model {args.model} is INCOMPLETE!")
            print(f"   Missing items:")
            if config_found < config_total:
                print(f"   - {config_total - config_found} config files")
            if not index:
                print(f"   - safetensors index")
            if index and not tensors_complete:
                print(f"   - Some tensor keys")

    finally:
        store.close()

if __name__ == '__main__':
    main()
