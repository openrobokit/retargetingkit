import os
import sys
import sysconfig
from pathlib import Path

import pytest

from retargetingkit.targets.hands import sharpa_hand


class TestSharpaSDKInstall:
    def _reset_sdk_state(self):
        sharpa_hand._sdk_initialized = False

    def _make_sdk_tree(self, root: Path, include_python_module: bool = True) -> Path:
        sdk_root = root / "SDK"
        ext_suffix = str(sysconfig.get_config_var("EXT_SUFFIX"))
        (sdk_root / "lib").mkdir(parents=True)
        (sdk_root / "python" / "sharpa").mkdir(parents=True)
        (sdk_root / "python" / "sharpa" / "__init__.py").write_text("from .sharpa import *\n")
        (sdk_root / "lib" / "libsharpa-wave-sdk.so").write_bytes(b"binary")
        (sdk_root / "lib" / "libSharpaWaveSDKWrapper.so").write_bytes(b"binary")
        if include_python_module:
            (sdk_root / "python" / "sharpa" / f"sharpa{ext_suffix}").write_bytes(b"binary")
        return sdk_root

    def test_init_sdk_uses_existing_install(self, monkeypatch, tmp_path: Path):
        sdk_path = tmp_path / "thirdparty" / "sharpa-sdk"
        existing_sdk = self._make_sdk_tree(tmp_path / "thirdparty")
        existing_sdk.rename(sdk_path)
        monkeypatch.setattr(sharpa_hand, "_SDK_PATH", sdk_path)
        monkeypatch.setenv("LD_LIBRARY_PATH", "")
        self._reset_sdk_state()
        sys.path = [entry for entry in sys.path if entry != str(sdk_path / "python")]

        sharpa_hand._init_sdk()

        assert sharpa_hand._sdk_initialized is True
        assert str(sdk_path / "python") in sys.path
        assert str(sdk_path / "lib") in os.environ["LD_LIBRARY_PATH"].split(":")
        assert sysconfig.get_config_var("LIBDIR") in os.environ["LD_LIBRARY_PATH"].split(":")

    def test_init_sdk_requires_current_python_module(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(sharpa_hand, "_SDK_PATH", self._make_sdk_tree(tmp_path, include_python_module=False))
        self._reset_sdk_state()

        with pytest.raises(AssertionError, match="Install the Sharpa SDK"):
            sharpa_hand._init_sdk()
