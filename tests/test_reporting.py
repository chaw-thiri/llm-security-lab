"""Documentation refactors must preserve scientific evidence and avoid rerunning studies."""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

from promptshield.provenance import ARCHIVE, frozen_protocol_sha256
from promptshield.reporting import ROOT, REPORTS, build_report, write_all_reports


class ReportIntegrityTests(unittest.TestCase):
    def test_frozen_protocol_hashes_and_completed_source_are_verifiable(self):
        archive=json.loads(ARCHIVE.read_text())
        for identifier,record in archive["protocols"].items():
            self.assertEqual(frozen_protocol_sha256(identifier),record["sha256"])
        for name,identifier in (("grounded-upgrade-data.json","docs/grounded-upgrade-protocol.md"),
                                ("context-consistency-data.json","docs/context-consistency-protocol.md")):
            self.assertEqual(json.loads((ROOT/"docs"/name).read_text())["protocol_sha256"],frozen_protocol_sha256(identifier))
        snapshot=json.loads((ROOT/"docs/provenance/context-consistency-execution.json").read_text())
        results=json.loads((ROOT/"docs/context-consistency-results.json").read_text())
        self.assertEqual(snapshot["execution_sha256"],results["execution_sha256"])
        for name,record in snapshot["files"].items():
            self.assertEqual(hashlib.sha256(record["content_utf8"].encode()).hexdigest(),record["sha256"],name)

    def test_protocol_tampering_fails_closed(self):
        archive=copy.deepcopy(json.loads(ARCHIVE.read_text()))
        key=next(iter(archive["protocols"]))
        archive["protocols"][key]["content_utf8"]+="Changed method"
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"archive.json"; path.write_text(json.dumps(archive))
            with self.assertRaises(ValueError):
                frozen_protocol_sha256(key,path)

    def test_reports_are_compact_and_leave_measured_json_unchanged(self):
        inputs={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"docs").glob("*.json")}
        for study in REPORTS:
            text=build_report(study)
            self.assertLessEqual(len(text.split()),650,study)
            self.assertLessEqual(len(text.splitlines()),60,study)
            self.assertNotRegex(text,r"(?im)^#{1,6}\s*(?:question|answer)\s*:")
            self.assertNotIn("**Answer:**",text)
        self.assertEqual(inputs,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
        write_all_reports(check=True)

    def test_public_markdown_links_resolve(self):
        paths=[ROOT/"README.md",ROOT/"contribution.md",ROOT/"research_diary.md"]+list((ROOT/"docs").glob("*.md"))
        for path in paths:
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)",path.read_text()):
                if target.startswith(("https://","http://","#","mailto:")):
                    continue
                target=target.split("#",1)[0]
                self.assertTrue((path.parent/target).exists(),f"{path.name}: {target}")

    def test_completed_training_entrypoints_render_only(self):
        sys.path.insert(0,str(ROOT/"scripts"))
        import run_context_consistency
        import run_grounded_upgrade
        for module in (run_context_consistency,run_grounded_upgrade):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); (root/"status.json").write_text('{"status":"complete"}')
                with patch.object(module,"OUT",root),patch.object(module,"write_compact_report") as render,patch.object(module,"train_arm",side_effect=AssertionError("Unexpected training")):
                    module.main()
                    render.assert_called_once()


if __name__=="__main__":
    unittest.main()
