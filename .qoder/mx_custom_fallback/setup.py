"""mx_custom_fallback 安装脚本。

安装方式:
    pip install -e .                    # 开发模式
    pip install .                       # 正常安装

安装后，设置环境变量 MX_FALLBACK_SOURCE_URL 即可激活自动注入。
包通过 .pth 文件在 Python 启动时自动 import，无需手动修改 sglang 代码。

如果不想自动 import（只想手动 import），设置环境变量:
    MX_FALLBACK_NO_AUTOIMPORT=1
"""

import os
import shutil
import site
from pathlib import Path

from setuptools import setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop


_PTH_CONTENT = (
    "import os; "
    "os.environ.get('MX_FALLBACK_NO_AUTOIMPORT', '0') != '1' "
    "and __import__('mx_custom_fallback')\n"
)


def _install_pth_file():
    """将 .pth 文件复制到 site-packages 目录以实现自动导入。"""
    for sp_dir in site.getsitepackages() + [site.getusersitepackages()]:
        pth_path = Path(sp_dir) / "mx_custom_fallback.pth"
        try:
            pth_path.write_text(_PTH_CONTENT)
            print(f"[mx_custom_fallback] Installed .pth to {pth_path}")
            return
        except (OSError, PermissionError):
            continue
    print("[mx_custom_fallback] WARNING: Could not install .pth file automatically.")
    print("  Manual setup: copy the following to a .pth file in your site-packages:")
    print(f"  {_PTH_CONTENT.strip()}")


class _CustomInstall(_install):
    def run(self):
        super().run()
        _install_pth_file()


class _CustomDevelop(_develop):
    def run(self):
        super().run()
        _install_pth_file()


setup(
    name="mx-custom-fallback",
    version="0.1.0",
    description="Zero-intrusion custom fallback source injection for ModelExpress + SGLang",
    packages=["mx_custom_fallback"],
    python_requires=">=3.9",
    install_requires=[],
    cmdclass={
        "install": _CustomInstall,
        "develop": _CustomDevelop,
    },
)
