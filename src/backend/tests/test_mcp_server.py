import json
import unittest

from claude_dj.mcp.server import mcp_json_result


class MCPServerTests(unittest.TestCase):
    def test_mcp_json_result_wraps_payload_as_text_content(self) -> None:
        result = mcp_json_result({"available": True, "candidates": [{"id": "track-1"}]})

        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(
            json.loads(result["content"][0]["text"]),
            {"available": True, "candidates": [{"id": "track-1"}]},
        )


if __name__ == "__main__":
    unittest.main()
