"""Exercise the release ZIP, including metadata and artifact exclusions."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


class ReleaseArchiveTest(unittest.TestCase):
    def test_release_includes_citation_and_ci_and_excludes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "release.zip"
            command = [sys.executable, str(ROOT / "scripts/package_release.py"),
                       "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True)
            with zipfile.ZipFile(output) as archive:
                names = {str(Path(name).relative_to("LeaP_release"))
                         for name in archive.namelist()}
                self.assertIn("CITATION.cff", names)
                self.assertIn(".github/workflows/tests.yml", names)
                self.assertIn("docs/assets/leap-overview.mp4", names)
                self.assertIn("third_party/robotwin.lock.json", names)
                self.assertIn("third_party/curobo.lock.json", names)
                self.assertFalse(any("__pycache__" in name or name.startswith(
                    (".git/", "data/", "outputs/", "third_party/RoboTwin/"))
                    for name in names))
                self.assertIsNone(archive.testzip())
            original = output.read_bytes()
            repeat = subprocess.run(command, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
