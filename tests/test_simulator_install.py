"""Check dependency pinning against a real local Git repository, without CUDA."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from install_robotwin import pinned_installer


class SimulatorInstallTest(unittest.TestCase):
    def test_installer_checks_out_locked_api_instead_of_default_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "upstream with spaces"
            source.mkdir()

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", str(source), *args], text=True,
                    stderr=subprocess.DEVNULL).strip()

            git("init", "-b", "main")
            git("config", "user.name", "Test")
            git("config", "user.email", "test@example.invalid")
            (source / "api.txt").write_text("legacy motion_gen")
            git("add", "."); git("commit", "-m", "legacy API")
            commit = git("rev-parse", "HEAD")
            (source / "api.txt").write_text("incompatible new API")
            git("commit", "-am", "new API")
            (root / "envs").mkdir()
            script = ('cd envs\ngit clone https://github.com/NVlabs/curobo.git\n'
                      'cd curobo\ncat api.txt\n')
            locked = pinned_installer(script, {"url": str(source), "commit": commit})
            result = subprocess.run(["bash", "-e", "-c", locked], cwd=root,
                                    check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "legacy motion_gen")
            self.assertEqual(subprocess.check_output(
                ["git", "-C", str(root / "envs/curobo"), "rev-parse", "HEAD"],
                text=True).strip(), commit)

    def test_unknown_installer_layout_is_rejected(self):
        with self.assertRaises(ValueError):
            pinned_installer("git clone unexpected-upstream\n",
                             {"url": "unused", "commit": "a" * 40})


if __name__ == "__main__":
    unittest.main()
