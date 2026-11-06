"""Negative controls for build provenance and executable documentation checks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReferenceRunnerTests(unittest.TestCase):
    def run_reference(self, text, build_dir=None, fallback=None):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.md"
            reference.write_text(text, encoding="utf-8")
            env = dict(os.environ)
            if build_dir is not None:
                env["PYNEURONET_BUILD_DIR"] = str(build_dir)
            if fallback is not None:
                env["PYTHONPATH"] = str(fallback)
            return subprocess.run(
                [sys.executable, "-O", str(Path(__file__).with_name("test_python_reference.py")),
                 str(reference)], env=env, capture_output=True, text=True, timeout=30)

    def test_missing_extension_rejects_available_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            # A usable extension remains on PYTHONPATH, outside the selected build.
            result = self.run_reference("```python\npass\n```\n", directory,
                                        os.environ["PYNEURONET_BUILD_DIR"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Matching build-tree", result.stderr)

    def test_source_module_is_not_a_build_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "pyneuronet.py").write_text(
                'raise RuntimeError("source substitute executed")', encoding="utf-8")
            result = self.run_reference("```python\npass\n```\n", directory)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Matching build-tree", result.stderr)
        self.assertNotIn("source substitute executed", result.stderr)

    def test_missing_example_is_rejected(self):
        result = self.run_reference("No executable example.\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No Python examples", result.stderr)

    def test_assertions_survive_optimized_python(self):
        result = self.run_reference("```python\nassert False, 'negative control'\n```\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AssertionError: negative control", result.stderr)


if __name__ == "__main__":
    unittest.main()
