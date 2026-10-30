from pathlib import Path
from unittest.mock import Mock

import pytest

from retargetingkit.utils.mano import ManoHand, ManoHandConfig, default_mano_root


class TestManoHand:
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_missing_model_gives_local_setup_instructions(
        self, side: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ROBOKIT_BODY_MODELS", str(tmp_path))
        download = Mock(side_effect=AssertionError("Tests must not access HF"))
        monkeypatch.setattr("huggingface_hub.snapshot_download", download)
        assert default_mano_root() == tmp_path / "mano"
        with pytest.raises(FileNotFoundError) as error:
            ManoHand(ManoHandConfig(side=side, device="cpu"))
        message = str(error.value)
        assert f"MANO_{side.upper()}" in message
        assert "ROBOKIT_BODY_MODELS" in message
        assert "https://mano.is.tue.mpg.de/" in message
        download.assert_not_called()
