"""Tests that drive the shipped MAS packaging files and launch entry."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path


class TestShippedBundleMetadata(unittest.TestCase):
    def test_info_plist_identity_from_shipped_file(self) -> None:
        from airllm_studio.bundle import APP_DISPLAY_NAME, BUNDLE_IDENTIFIER, load_info_plist

        info = load_info_plist()
        self.assertEqual(info.get("CFBundleIdentifier"), BUNDLE_IDENTIFIER)
        self.assertEqual(info.get("CFBundleDisplayName"), APP_DISPLAY_NAME)
        self.assertTrue(info.get("CFBundleShortVersionString"))
        self.assertTrue(info.get("CFBundleVersion"))
        self.assertEqual(info.get("CFBundleExecutable"), "AirLLMStudio")
        self.assertTrue(info.get("CFBundleName"))
        copyright_ = str(info.get("NSHumanReadableCopyright") or "")
        self.assertIn("2026", copyright_)
        self.assertIn("Michael Beck", copyright_)
        self.assertIn("All rights reserved", copyright_)

    def test_entitlements_declare_sandbox_network_and_files(self) -> None:
        from airllm_studio.bundle import load_entitlements

        ents = load_entitlements()
        self.assertIs(ents.get("com.apple.security.app-sandbox"), True)
        self.assertIs(ents.get("com.apple.security.network.client"), True)
        self.assertIs(ents.get("com.apple.security.files.user-selected.read-write"), True)
        # Hardened Runtime exceptions required for CPython + PyTorch/MLX
        self.assertIs(ents.get("com.apple.security.cs.allow-jit"), True)
        self.assertIs(ents.get("com.apple.security.cs.allow-unsigned-executable-memory"), True)

    def test_signing_config_requests_hardened_runtime(self) -> None:
        from airllm_studio.bundle import load_signing_config

        cfg = load_signing_config()
        self.assertTrue(cfg.get("hardened_runtime"))
        opts = cfg.get("codesign_options") or []
        self.assertIn("runtime", opts)

    def test_bundle_identity_helper_reads_same_plist(self) -> None:
        from airllm_studio.bundle import BUNDLE_IDENTIFIER, bundle_identity, load_info_plist

        ident = bundle_identity()
        raw = load_info_plist()
        self.assertEqual(ident["bundle_identifier"], raw["CFBundleIdentifier"])
        self.assertEqual(ident["bundle_identifier"], BUNDLE_IDENTIFIER)
        self.assertIn("AirLLM", ident["display_name"])


class TestPackagedLaunchEntry(unittest.TestCase):
    def test_help_names_airllm_studio(self) -> None:
        from airllm_studio.launch import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        out = buf.getvalue()
        self.assertIn("AirLLM Studio", out)

    def test_version_prints_bundle_id(self) -> None:
        from airllm_studio.launch import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--version"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("AirLLM Studio", out)
        self.assertIn("ai.airllm.studio", out)

    def test_smoke_applies_writable_data_dir(self) -> None:
        from airllm_studio.launch import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--smoke"])
        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("AirLLM Studio smoke OK", out)
        self.assertIn("ai.airllm.studio", out)

    def test_sandbox_env_points_hf_cache_under_application_support(self) -> None:
        import os

        from airllm_studio.runtime_env import apply_sandbox_environment

        root = apply_sandbox_environment()
        self.assertTrue(root.exists())
        hf = Path(os.environ["HF_HOME"])
        self.assertTrue(str(hf).startswith(str(root)))
        self.assertTrue(hf.exists())
        from airllm_studio.runtime_env import assert_data_dir_writable

        assert_data_dir_writable(root)


class TestAppBundleBuild(unittest.TestCase):
    def test_build_app_writes_plist_and_launcher(self) -> None:
        import os
        import plistlib
        import tempfile

        from airllm_studio.packaging import APP_BUNDLE_NAME, build_app

        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / APP_BUNDLE_NAME
            built = build_app(dest)
            self.assertTrue(built.is_dir())
            plist_path = built / "Contents" / "Info.plist"
            exe = built / "Contents" / "MacOS" / "AirLLMStudio"
            ents = built / "Contents" / "Resources" / "AirLLMStudio.entitlements"
            pkg = built / "Contents" / "Resources" / "pythonpath" / "airllm_studio" / "launch.py"
            self.assertTrue(plist_path.is_file())
            self.assertTrue(exe.is_file())
            self.assertTrue(os.access(exe, os.X_OK))
            self.assertTrue(ents.is_file())
            self.assertTrue(pkg.is_file())
            with plist_path.open("rb") as fh:
                info = plistlib.load(fh)
            self.assertEqual(info["CFBundleIdentifier"], "ai.airllm.studio")
            launcher = exe.read_text(encoding="utf-8")
            self.assertIn("airllm_studio.launch", launcher)
            self.assertIn("PYTHONPATH", launcher)
            sites = built / "Contents" / "Resources" / "site-packages.txt"
            self.assertTrue(sites.is_file())
            storekit = built / "Contents" / "Resources" / "Products.storekit"
            self.assertTrue(storekit.is_file())
            launcher = exe.read_text(encoding="utf-8")
            self.assertNotIn("AIRLLM_STUDIO_RELEASE=1", launcher)
            release_dest = Path(tmp) / "Release.app"
            release_built = build_app(release_dest, release=True)
            rel_launcher = (release_built / "Contents" / "MacOS" / "AirLLMStudio").read_text(
                encoding="utf-8"
            )
            self.assertIn("AIRLLM_STUDIO_RELEASE=1", rel_launcher)


if __name__ == "__main__":
    unittest.main()
