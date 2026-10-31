import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("threshold_analysis",Path(__file__).resolve().parents[1]/"scripts/threshold_analysis.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class ThresholdAnalysisTests(unittest.TestCase):
    def test_evaluation_labels_cannot_change_selected_threshold(self):
        validation=[{"label":0,"score":.1,"source":"original"},{"label":1,"score":.8,"source":"original"},{"label":0,"score":.9,"source":"context-validation"},{"label":1,"score":.95,"source":"context-validation"}]
        a=module.analyze(validation,{"challenge":[{"label":0,"score":.7},{"label":1,"score":.6}]},[0])
        b=module.analyze(validation,{"challenge":[{"label":1,"score":.7},{"label":0,"score":.6}]},[0])
        self.assertEqual([p["threshold"] for p in a],[p["threshold"] for p in b])
        self.assertNotEqual(a[0]["threshold"],a[1]["threshold"])
        for policy in a:self.assertLessEqual(policy["validation"]["benign_fpr"],policy["fpr_budget"])
