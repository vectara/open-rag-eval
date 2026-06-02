import unittest

from omegaconf import OmegaConf

from open_rag_eval.chunking import ChunkingStrategy, parse_chunking_strategies


class TestChunkingStrategy(unittest.TestCase):
    def test_parse_basic_strategies(self):
        cfg = OmegaConf.create({
            "strategies": [
                {"name": "small", "chunk_size": 256, "chunk_overlap": 32},
                {"name": "large", "chunk_size": 1024, "chunk_overlap": 128},
            ]
        })
        strategies = parse_chunking_strategies(cfg)
        self.assertEqual(len(strategies), 2)
        self.assertIsInstance(strategies[0], ChunkingStrategy)
        self.assertEqual(strategies[0].name, "small")
        self.assertEqual(strategies[0].chunk_size, 256)
        self.assertEqual(strategies[0].chunk_overlap, 32)
        self.assertEqual(strategies[0].splitter, "recursive")
        self.assertEqual(strategies[1].name, "large")

    def test_parse_plain_dict(self):
        cfg = {"strategies": [{"name": "a", "chunk_size": 100, "chunk_overlap": 10}]}
        strategies = parse_chunking_strategies(cfg)
        self.assertEqual(len(strategies), 1)
        self.assertEqual(strategies[0].name, "a")

    def test_defaults_applied(self):
        cfg = {"strategies": [{"name": "defaulted"}]}
        strategies = parse_chunking_strategies(cfg)
        self.assertEqual(strategies[0].chunk_size, 1000)
        self.assertEqual(strategies[0].chunk_overlap, 200)

    def test_empty_strategies_raises(self):
        with self.assertRaises(ValueError):
            parse_chunking_strategies({"strategies": []})

    def test_missing_strategies_raises(self):
        with self.assertRaises(ValueError):
            parse_chunking_strategies({})

    def test_none_config_raises(self):
        with self.assertRaises(ValueError):
            parse_chunking_strategies(None)

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError):
            parse_chunking_strategies({"strategies": [{"chunk_size": 100}]})

    def test_duplicate_name_raises(self):
        cfg = {"strategies": [{"name": "dup"}, {"name": "dup"}]}
        with self.assertRaises(ValueError):
            parse_chunking_strategies(cfg)

    def test_overlap_ge_size_raises(self):
        cfg = {"strategies": [{"name": "bad", "chunk_size": 100, "chunk_overlap": 100}]}
        with self.assertRaises(ValueError):
            parse_chunking_strategies(cfg)

    def test_non_positive_size_raises(self):
        cfg = {"strategies": [{"name": "bad", "chunk_size": 0, "chunk_overlap": 0}]}
        with self.assertRaises(ValueError):
            parse_chunking_strategies(cfg)

    def test_negative_overlap_raises(self):
        cfg = {"strategies": [{"name": "bad", "chunk_size": 100, "chunk_overlap": -5}]}
        with self.assertRaises(ValueError):
            parse_chunking_strategies(cfg)


if __name__ == "__main__":
    unittest.main()
