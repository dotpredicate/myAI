import unittest

from inference import llama_cpp_server as module
from inference.engine import TokenPiece


class TestTokenizationResponse(unittest.TestCase):
    def test_parses_text_and_byte_pieces(self) -> None:
        result = module._parse_tokenization_response({
            "tokens": [
                {"id": 1, "piece": "hello"},
                {"id": 2, "piece": [195, 169]},
            ]
        })

        self.assertEqual(result, [TokenPiece(1, "hello"), TokenPiece(2, [195, 169])])

    def test_rejects_malformed_token(self) -> None:
        with self.assertRaises(ValueError):
            module._parse_tokenization_response({"tokens": [{"id": 1, "piece": [256]}]})


if __name__ == "__main__":
    unittest.main()
