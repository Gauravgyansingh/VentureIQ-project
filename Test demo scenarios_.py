import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import demo_scenarios as d
except ModuleNotFoundError as e:           # fastapi not installed in this environment
    d = None; _err = e

@unittest.skipIf(d is None, "fastapi not installed")
class DemoTests(unittest.TestCase):
    def test_five_scenarios_complete_and_labelled(self):
        self.assertEqual(len(d.SCENARIOS), 5)
        need = {"concept","target_customers","business_model","market_assumptions","key_risks","key_opportunities","swot","recommendations","predict_inputs","explore_inputs"}
        for s in d.SCENARIOS:
            self.assertTrue(need <= s.keys(), s["id"])
            self.assertTrue(s["illustrative"]); self.assertIn("ILLUSTRATIVE", s["disclaimer"])
            self.assertEqual(set(s["swot"]), {"strengths","weaknesses","opportunities","threats"})
            self.assertNotIn("viability_score", s)       # no invented scores
    def test_predict_inputs_match_api_constraints(self):
        for s in d.SCENARIOS:
            p = s["predict_inputs"]
            self.assertTrue(p["total_funding_usd"] >= 0 and p["funding_rounds"] >= 0)
            self.assertTrue(all(p[k] in (0,1) for k in ("has_vc","has_angel","has_series_a","has_series_b")))
            if p["has_series_b"]: self.assertTrue(p["has_series_a"])
if __name__ == "__main__": unittest.main()