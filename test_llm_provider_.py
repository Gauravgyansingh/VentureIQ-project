"""Provider selection and client behaviour, using a fake HTTP session (no network)."""
import os, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from groq_helper import GroqClient, LLMAuthError, LLMQuotaError, LLMResponseError
from utils import redact

FAKE_KEY = "sk-or-v1-" + "a1b2c3d4" * 6


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
    def json(self):
        if self._b is None: raise ValueError("no json")
        return self._b


class Sess:
    def __init__(self, *resps): self.resps, self.calls = list(resps), []
    def post(self, url, **kw): self.calls.append((url, kw)); return self.resps.pop(0)
    def get(self, url, **kw): self.calls.append((url, kw)); return self.resps.pop(0)


def ok(text): return Resp(200, {"choices": [{"message": {"content": text}}]})
ident = lambda d: d


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self._env = dict(os.environ)
        for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDER", "OPENROUTER_MODEL"):
            os.environ.pop(k, None)
    def tearDown(self):
        os.environ.clear(); os.environ.update(self._env)

    def test_openrouter_selected_from_env(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        os.environ["OPENROUTER_MODEL"] = "openrouter/free"
        c = GroqClient.from_settings(session=Sess())
        self.assertEqual((c.provider, c.model), ("openrouter", "openrouter/free"))
        self.assertIn("openrouter.ai", c.base_url)

    def test_groq_still_works(self):
        os.environ["GROQ_API_KEY"] = "gsk_" + "x" * 20
        c = GroqClient.from_settings(session=Sess())
        self.assertEqual(c.provider, "groq")

    def test_no_key_means_none(self):
        self.assertIsNone(GroqClient.from_settings())

    def test_openrouter_request_shape(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        s = Sess(ok('{"a": 1}'))
        out = GroqClient.from_settings(session=s).chat_json("sys", "usr", ident)
        self.assertEqual(out, {"a": 1})
        url, kw = s.calls[0]
        self.assertTrue(url.endswith("/chat/completions"))
        self.assertNotIn("response_format", kw["json"])
        self.assertEqual(kw["headers"]["X-Title"], "VentureIQ")

    def test_fenced_json_and_retry_on_invalid(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        s = Sess(ok("not json at all"), ok("```json\n{\"b\": 2}\n```"))
        self.assertEqual(GroqClient.from_settings(session=s).chat_json("s", "u", ident), {"b": 2})
        self.assertEqual(len(s.calls), 2)

    def test_errors_map_to_clear_exceptions(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        for status, exc in ((401, LLMAuthError), (402, LLMQuotaError), (429, LLMQuotaError)):
            with self.assertRaises(exc):
                GroqClient.from_settings(session=Sess(Resp(status))).chat_json("s", "u", ident)

    def test_empty_content_is_response_error(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        with self.assertRaises(LLMResponseError):
            GroqClient.from_settings(session=Sess(Resp(200, {"choices": [{"message": {"content": None}}]}))).chat_json("s", "u", ident)

    def test_ping_uses_authenticated_endpoint(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        s = Sess(Resp(200, {}))
        self.assertTrue(GroqClient.from_settings(session=s).ping()[0])
        self.assertTrue(s.calls[0][0].endswith("/key"))

    def test_redaction_covers_openrouter_keys(self):
        os.environ["OPENROUTER_API_KEY"] = FAKE_KEY
        self.assertNotIn(FAKE_KEY, redact(f"boom {FAKE_KEY} boom"))
        self.assertIn("[REDACTED]", redact("Bearer sk-or-v1-abcdef123456789"))


if __name__ == "__main__":
    unittest.main()