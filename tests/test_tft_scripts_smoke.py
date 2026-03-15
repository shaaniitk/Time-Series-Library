import subprocess
import sys
from pathlib import Path
import unittest
import re


class TestTFTScriptsSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _run_script(self, rel_path, args, timeout=240):
        cmd = [sys.executable, str(self.repo_root / rel_path)] + args
        completed = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"Script failed: {' '.join(cmd)}\n"
                f"Return code: {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )
        return completed.stdout + completed.stderr

    def test_dummy_script_quick(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_dummy_40cov_4target_example.py"),
            ["--quick", "--device", "cpu", "--seed", "123"],
        )
        self.assertIn("predictions_full shape:", output)
        self.assertIn("attention_weights shape:", output)

    def test_ablation_script_quick(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_ablation_full_attention_vs_vsn_bypass.py"),
            ["--quick", "--device", "cpu", "--seeds", "123,456"],
        )
        self.assertIn("Ablation: TFT dependency-upgrade matrix", output)
        self.assertIn("Best config by mean loss", output)
        self.assertIn("all_upgrades", output)
        self.assertIn("graph", output)
        self.assertIn("interaction", output)
        self.assertIn("moe", output)

    def test_ablation_overfit_signal_present(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_ablation_full_attention_vs_vsn_bypass.py"),
            ["--quick", "--device", "cpu", "--seeds", "123"],
        )
        # Ensure at least one configuration can overfit a single batch enough to pass the sanity check.
        self.assertRegex(output, re.compile(r"\byes\b"))
        self.assertIn("overfit_ratio", output)

    def test_cross_revin_regime_ablation_quick(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_ablation_cross_revin_regime.py"),
            ["--quick", "--device", "cpu", "--seeds", "123"],
            timeout=300,
        )
        self.assertIn("Ablation: TFT cross-attention + RevIN + timestep regime routing", output)
        self.assertIn("all_three", output)
        self.assertIn("cross_only", output)
        self.assertIn("revin_only", output)
        self.assertIn("regime_only", output)

    def test_attention_backend_benchmark_quick(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_attention_backend_benchmark.py"),
            ["--quick", "--device", "cpu"],
            timeout=300,
        )
        self.assertIn("Benchmark: TFT exact vs SDPA attention backend", output)
        self.assertIn("exact", output)
        self.assertIn("sdpa", output)

    def test_backbone_harsh_ablation_quick(self):
        output = self._run_script(
            Path("scripts/long_term_forecast/tft_backbone_harsh_signal_ablation.py"),
            ["--quick", "--device", "cpu", "--seeds", "123"],
            timeout=300,
        )
        self.assertIn("Ablation: TFT temporal backbone on harsh synthetic signal", output)
        self.assertIn("lstm", output)
        self.assertIn("gated_tcn", output)
        self.assertIn("hybrid_tcn_lstm", output)


if __name__ == "__main__":
    unittest.main()
