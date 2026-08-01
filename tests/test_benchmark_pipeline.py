import unittest
from pathlib import Path

from benchmarks.retrieval import _is_cuda_oom, compare_retrieval_answer
from benchmarks.base_tasks import _choice_example
from benchmarks.longdoc import _parse_length
from benchmarks.aggregate import _flatten
from benchmarks.run_pipeline import (
    MODEL_SPECS,
    BenchmarkJob,
    _job_fingerprint,
    _jobs,
    _parser,
)
from benchmarks.scoring import DirectScorer, TokenRequest, encode_pair


class RetrievalScoringTest(unittest.TestCase):
    def test_integer_key_accepts_harmless_prose(self):
        self.assertTrue(compare_retrieval_answer("The pass key is 1234567.", "1234567"))

    def test_integer_key_rejects_substring_or_wrong_key(self):
        self.assertFalse(compare_retrieval_answer("The pass key is 12345678.", "1234567"))
        self.assertFalse(compare_retrieval_answer("7654321", "1234567"))

    def test_oom_classifier_does_not_hide_generic_runtime_errors(self):
        self.assertTrue(_is_cuda_oom(RuntimeError("CUDA out of memory")))
        self.assertFalse(_is_cuda_oom(RuntimeError("kernel launch failed")))


class PipelineFingerprintTest(unittest.TestCase):
    def _job(self, **changes):
        values = dict(
            stage="full",
            benchmark="base",
            suite="zero_shot",
            model="polar",
            checkpoint_revision="0123456789abcdef",
            dataset_revisions=(("Rowan/hellaswag", "abcdef"),),
            command_args=("--tasks", "hellaswag", "--batch_size", "8"),
        )
        values.update(changes)
        return BenchmarkJob(**values)

    def test_fingerprint_is_stable_and_configuration_sensitive(self):
        self.assertEqual(_job_fingerprint(self._job()), _job_fingerprint(self._job()))
        self.assertNotEqual(
            _job_fingerprint(self._job()),
            _job_fingerprint(self._job(command_args=("--tasks", "hellaswag", "--batch_size", "4"))),
        )

    def test_full_stage_contains_every_automatic_benchmark_for_every_model(self):
        from benchmarks.base_tasks import BASE_TASK_SPECS
        from benchmarks.longdoc import LONGDOC_SPECS

        args = _parser().parse_args(["--stage", "full"])
        args.checkpoint_revisions = {name: f"sha-{name}" for name in MODEL_SPECS}
        dataset_ids = {
            spec.dataset_id for spec in (*BASE_TASK_SPECS.values(), *LONGDOC_SPECS.values())
        }
        dataset_ids.add(args.haystack)
        args.dataset_revisions = {name: "dataset-sha" for name in dataset_ids}
        args.manifest_path = Path("checkpoint_manifest.json")

        jobs = _jobs(args)
        counts = {
            benchmark: sum(job.benchmark == benchmark for job in jobs)
            for benchmark in {job.benchmark for job in jobs}
        }
        self.assertEqual(
            counts,
            {"retrieval": 10, "base": 5, "longdoc": 5, "serving": 5},
        )


class AggregateTest(unittest.TestCase):
    def test_base_result_is_flattened(self):
        rows = _flatten(
            "base",
            {
                "model_config": {"attn_type": "polar"},
                "results": {"hellaswag": {"samples": 10, "accuracy_norm": 0.7}},
            },
            Path("base.log"),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metric"], "accuracy_norm")
        self.assertEqual(rows[0]["value"], 0.7)


class TaskFormattingTest(unittest.TestCase):
    def test_boolq_and_winogrande_choices(self):
        context, choices, gold = _choice_example(
            "boolq", {"passage": "Water is wet.", "question": "is water wet", "answer": True}
        )
        self.assertTrue(context.endswith("Answer:"))
        self.assertEqual((choices, gold), (["no", "yes"], 1))

        context, choices, gold = _choice_example(
            "winogrande",
            {"sentence": "The trophy does not fit because _ is large.",
             "option1": "the trophy", "option2": "the case", "answer": "1"},
        )
        self.assertEqual(context, "The trophy does not fit because")
        self.assertEqual(choices[0], "the trophy is large.")
        self.assertEqual(gold, 0)

        context, choices, gold = _choice_example(
            "hellaswag",
            {
                "activity_label": "Baking cookies",
                "ctx_a": "A baker opens the oven.",
                "ctx_b": "they",
                "endings": ["remove the tray.", "turn off the rain."],
                "label": "0",
            },
        )
        self.assertEqual(context, "Baking cookies: A baker opens the oven. They")
        self.assertEqual(choices[0], "remove the tray.")
        self.assertEqual(gold, 0)

    def test_pair_encoding_preserves_boundary_space(self):
        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        request = encode_pair(CharacterTokenizer(), "hello ", "world")
        self.assertEqual("".join(map(chr, request.context_ids)), "hello")
        self.assertEqual("".join(map(chr, request.continuation_ids)), " world")

    def test_length_parser(self):
        self.assertEqual(_parse_length("256k"), 262144)


class DirectScoringTest(unittest.TestCase):
    def test_continuation_alignment_matches_next_token_cross_entropy(self):
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch is not installed in the lightweight development environment")

        class ToyModel:
            def __init__(self):
                self.embed = torch.nn.Embedding(6, 6)
                self.blocks = []
                self.norm = torch.nn.Identity()
                self.proj = torch.nn.Linear(6, 6, bias=False)
                with torch.no_grad():
                    self.embed.weight.copy_(torch.eye(6))
                    self.proj.weight.copy_(torch.eye(6))

        scorer = object.__new__(DirectScorer)
        scorer.model = ToyModel()
        scorer.device = torch.device("cpu")
        scorer.max_length = 16
        scorer.tokenizer = type("Tokenizer", (), {"eos_token_id": 0})()

        result = scorer._score_batch([TokenRequest((1, 2), (3, 4))])[0]
        raw_logits = torch.eye(6)[torch.tensor([2, 3])]
        logits = 15.0 * raw_logits * (raw_logits.square() + 15.0**2).rsqrt()
        expected = F.cross_entropy(logits, torch.tensor([3, 4]), reduction="sum")
        self.assertAlmostEqual(result["loglikelihood"], -expected.item(), places=6)
        self.assertEqual(result["tokens"], 2)

if __name__ == "__main__":
    unittest.main()
