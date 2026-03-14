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


if __name__ == "__main__":
    unittest.main()
