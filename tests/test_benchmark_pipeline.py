import json
import unittest
import random
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmarks import retrieval as retrieval_module
from benchmarks.babilong import EVAL_LENGTHS, TRAIN_LENGTHS, run_babilong, select_row_ids
from benchmarks.finetune_babilong import encode_example, validate_protocol
from benchmarks.retrieval import (
    _is_cuda_oom,
    compare_retrieval_answer,
    make_sample,
    run_retrieval,
)
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
from benchmarks.run_babilong_pipeline import (
    _checkpoint_matches as _babilong_checkpoint_matches,
    _eval_command as _babilong_eval_command,
    _finetune_command as _babilong_finetune_command,
    _parser as _babilong_pipeline_parser,
    _repo_id as _babilong_repo_id,
    _upload_fingerprint as _babilong_upload_fingerprint,
    _write_upload_artifacts as _write_babilong_upload_artifacts,
)
from benchmarks.run import _infer_max_model_len
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

    def test_teacher_forced_sample_is_exact_length_and_reports_token_accuracy(self):
        class CharacterTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return list(text.encode("utf-8"))

        class PerfectScorer:
            def __init__(self):
                self.clears = 0

            def score_token_ids(self, context_ids, target_ids):
                count = len(target_ids)
                return {
                    "correct_tokens": count,
                    "tokens": count,
                    "greedy_exact": True,
                    "loglikelihood": -0.25 * count,
                }

            def clear_cache(self):
                self.clears += 1

        previous = retrieval_module._TOK
        retrieval_module._TOK = CharacterTokenizer()
        try:
            context, target = make_sample(
                "passkey", 2048, 0.5, random.Random(1234), value_tokens=5
            )
            self.assertEqual(len(context), 2048)
            self.assertGreater(len(target), 0)

            scorer = PerfectScorer()
            result = run_retrieval(
                scorer, ["passkey"], ["2k"], [0.5], num_samples=2, value_tokens=5
            )
            self.assertEqual(result["protocol"], "teacher-forced-needle-v2")
            self.assertEqual(result["results"]["passkey"]["2k"]["0.5"], 100.0)
            self.assertEqual(result["exact_results"]["passkey"]["2k"]["0.5"], 100.0)
            self.assertEqual(scorer.clears, 2)
        finally:
            retrieval_module._TOK = previous


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
        serving = next(job for job in jobs if job.benchmark == "serving")
        self.assertIn("128k", serving.command_args)
        self.assertNotIn("256k", serving.command_args)
        counts = {
            benchmark: sum(job.benchmark == benchmark for job in jobs)
            for benchmark in {job.benchmark for job in jobs}
        }
        self.assertEqual(
            counts,
            {"retrieval": 10, "base": 5, "longdoc": 5, "serving": 5},
        )


class BabilongPipelineTest(unittest.TestCase):
    def _args(self):
        args = _babilong_pipeline_parser().parse_args([])
        args.pinned_dataset_revision = "dataset-sha"
        return args

    def test_defaults_cover_every_checkpoint_and_256k(self):
        args = self._args()
        self.assertEqual(args.models, list(MODEL_SPECS))
        self.assertEqual(tuple(args.train_lengths), ("0k", "1k", "2k"))
        self.assertEqual(args.eval_lengths[-1], "256k")
        self.assertFalse(args.upload)
        self.assertEqual(
            _babilong_repo_id("ChavyvAkvar", "atma-10b-babilong-2k-ft", "raven_native"),
            "ChavyvAkvar/atma-10b-babilong-2k-ft-raven-native",
        )

    def test_subprocess_commands_pin_sources_dataset_and_eval_backend(self):
        args = self._args()
        source = Path("/tmp/source-checkpoint")
        output = Path("/tmp/babilong-checkpoint")
        train = _babilong_finetune_command(
            args,
            source,
            output,
            "owner/source",
            "source-sha",
        )
        self.assertIn("dataset-sha", train)
        self.assertIn("source-sha", train)
        self.assertEqual(
            train[train.index("--train_lengths") + 1:train.index("--seq_len")],
            ["0k", "1k", "2k"],
        )

        pilot = _babilong_eval_command(
            args,
            output,
            Path("/tmp/pilot.log"),
            pilot=True,
        )
        self.assertEqual(pilot[pilot.index("--tasks") + 1], "qa1")
        self.assertEqual(pilot[pilot.index("--lengths") + 1], "256k")
        self.assertEqual(pilot[pilot.index("--samples") + 1], "1")
        self.assertEqual(
            pilot[pilot.index("--babilong_backend") + 1],
            "direct",
        )

        full = _babilong_eval_command(
            args,
            output,
            Path("/tmp/full.log"),
            pilot=False,
        )
        lengths = full[full.index("--lengths") + 1:full.index("--row_start")]
        self.assertEqual(lengths[-1], "256k")
        self.assertIn("4k", lengths)
        self.assertIn("16k", lengths)

    def test_checkpoint_resume_requires_exact_protocol(self):
        args = self._args()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            (output / "weights.pt").write_bytes(b"weights")
            (output / "config.json").write_text("{}", encoding="utf-8")
            (output / "training_summary.json").write_text("{}", encoding="utf-8")
            manifest = {
                "protocol": "heldout-short-finetune-v1",
                "source_checkpoint": str(source.resolve()),
                "source_repo_id": "owner/source",
                "source_revision": "source-sha",
                "dataset_id": args.dataset,
                "dataset_revision": "dataset-sha",
                "tasks": list(args.tasks),
                "train_lengths": list(args.train_lengths),
                "seq_len": 2048,
                "train_rows": [0, 80],
                "validation_rows": [80, 90],
                "reserved_test_rows": [90, 100],
            }
            manifest_path = output / "finetune_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            kwargs = dict(
                source_repo="owner/source",
                source_revision="source-sha",
                dataset=args.dataset,
                dataset_revision="dataset-sha",
                tasks=args.tasks,
                train_lengths=args.train_lengths,
                seq_len=2048,
                train_rows=(0, 80),
                validation_rows=(80, 90),
                test_rows=(90, 100),
            )
            self.assertTrue(_babilong_checkpoint_matches(output, **kwargs))
            kwargs["dataset_revision"] = "different-sha"
            with self.assertRaisesRegex(RuntimeError, "protocol mismatch"):
                _babilong_checkpoint_matches(output, **kwargs)

    def test_upload_fingerprint_does_not_depend_on_previous_upload_record(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "weights.pt").write_bytes(b"weights")
            (output / "training_summary.json").write_text("{}", encoding="utf-8")
            (output / "finetune_manifest.json").write_text(
                json.dumps({
                    "dataset_id": "RMT-team/babilong",
                    "dataset_revision": "dataset-sha",
                    "source_revision": "source-sha",
                    "prompt_protocol": "builtin-v1",
                    "train_lengths": ["0k", "1k", "2k"],
                    "seq_len": 2048,
                    "train_rows": [0, 80],
                    "validation_rows": [80, 90],
                    "reserved_test_rows": [90, 100],
                }),
                encoding="utf-8",
            )
            stages = {
                "pilot": {"status": "complete", "result": {"benchmark": "babilong"}},
                "full_eval": {"status": "complete", "result": {"benchmark": "babilong"}},
                "upload": {"status": "complete", "commit": "first"},
            }
            kwargs = dict(
                spec=MODEL_SPECS["polar"],
                source_record={"resolved_revision": "source-sha"},
                target_repo="owner/target",
                dataset_record={
                    "dataset_id": "RMT-team/babilong",
                    "resolved_revision": "dataset-sha",
                },
                stages=stages,
            )
            _write_babilong_upload_artifacts(output, **kwargs)
            first = _babilong_upload_fingerprint(output)
            stages["upload"] = {
                "status": "complete",
                "commit": "second",
                "uploaded_at_unix": 123,
            }
            _write_babilong_upload_artifacts(output, **kwargs)
            second = _babilong_upload_fingerprint(output)
            self.assertEqual(first, second)


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

    def test_teacher_forced_retrieval_metrics_are_flattened_separately(self):
        result = {
            "protocol": "teacher-forced-needle-v2",
            "model_config": {"attn_type": "nope"},
            "haystack": "synthetic-filler",
            "num_samples": 2,
            "results": {"passkey": {"2k": {"0.5": 80.0}}},
            "exact_results": {"passkey": {"2k": {"0.5": 50.0}}},
            "nll_results": {"passkey": {"2k": {"0.5": 0.25}}},
        }
        rows = _flatten("retrieval", result, Path("retrieval.log"))
        self.assertEqual(
            {row["metric"] for row in rows},
            {"token_accuracy", "exact_match", "nll_nats_per_token"},
        )

    def test_legacy_generation_retrieval_is_not_aggregated(self):
        rows = _flatten(
            "retrieval",
            {
                "model_config": {"attn_type": "nope"},
                "results": {"passkey": {"2k": {"0.5": 0.0}}},
            },
            Path("legacy_retrieval.log"),
        )
        self.assertEqual(rows, [])

    def test_controlled_babilong_results_and_oom_are_flattened(self):
        rows = _flatten(
            "babilong",
            {
                "protocol": "heldout-short-finetune-v1",
                "model_config": {"attn_type": "polar"},
                "dataset_id": "RMT-team/babilong",
                "num_samples": 10,
                "results": {
                    "qa1": {"128k": 80.0, "256k": None},
                },
                "counts": {
                    "qa1": {"128k": 10, "256k": 10},
                },
                "oom_cells": [
                    {"task": "qa1", "length": "256k", "error": "out of memory"},
                ],
            },
            Path("babilong.log"),
        )
        self.assertEqual(
            [(row["length"], row["metric"], row["value"]) for row in rows],
            [("128k", "accuracy", 80.0), ("256k", "oom", True)],
        )

    def test_legacy_babilong_result_is_not_aggregated(self):
        rows = _flatten(
            "babilong",
            {
                "model_config": {"attn_type": "polar"},
                "results": {"qa1": {"2k": 0.0}},
            },
            Path("legacy_babilong.log"),
        )
        self.assertEqual(rows, [])


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
        self.assertEqual(_parse_length("4k"), 4096)
        self.assertEqual(_parse_length("16k"), 16384)
        self.assertEqual(_parse_length("256k"), 262144)


class BabilongProtocolTest(unittest.TestCase):
    class FakeDataset:
        def __init__(self, rows):
            self.rows = list(rows)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

        def select(self, row_ids):
            return self.__class__([self.rows[index] for index in row_ids])

        def __iter__(self):
            return iter(self.rows)

    def test_short_train_and_262k_eval_invariants(self):
        self.assertEqual(TRAIN_LENGTHS, ("0k", "1k", "2k"))
        self.assertEqual(EVAL_LENGTHS[-1], "256k")
        self.assertEqual(_infer_max_model_len(["256k"], 16), 262224)
        with self.assertRaisesRegex(ValueError, "subset"):
            validate_protocol(
                ["2k", "4k"],
                seq_len=2048,
                train_start=0,
                train_end=80,
                val_start=80,
                val_end=90,
            )
        with self.assertRaisesRegex(ValueError, "2048"):
            validate_protocol(
                ["0k", "1k", "2k"],
                seq_len=2049,
                train_start=0,
                train_end=80,
                val_start=80,
                val_end=90,
            )
        with self.assertRaisesRegex(ValueError, "reserved for test"):
            validate_protocol(
                ["0k", "1k", "2k"],
                seq_len=2048,
                train_start=0,
                train_end=80,
                val_start=80,
                val_end=91,
            )

    def test_test_rows_are_held_out_and_stable(self):
        dataset = self.FakeDataset([{"row": index} for index in range(100)])
        rows, row_ids = select_row_ids(
            dataset, row_start=90, row_end=100, num_samples=10
        )
        self.assertEqual(row_ids, list(range(90, 100)))
        self.assertEqual([row["row"] for row in rows], row_ids)

    def test_answer_only_labels_mask_prompt_and_padding(self):
        class CharacterTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [ord(char) for char in text]

        example = encode_example(
            CharacterTokenizer(),
            task="qa1",
            length="2k",
            row_id=3,
            row={"input": "Mary went to the kitchen.",
                 "question": "Where is Mary?", "target": "kitchen"},
            seq_len=512,
            official_prompts=None,
        )
        active = [token for token in example.labels if token != -100]
        self.assertEqual(active[-1], 0)
        self.assertGreater(len(active), 1)
        self.assertTrue(all(token == -100 for token in example.labels[-10:]))

    def test_eval_uses_only_held_out_rows(self):
        rows = [
            {"input": f"context {index}", "question": "Where?", "target": "kitchen"}
            for index in range(100)
        ]

        class PerfectModel:
            cfg = {"attn_type": "polar"}
            wip = []

            def __init__(self):
                self.clears = 0

            def generate(self, prompts, max_tokens=16):
                return ["kitchen"] * len(prompts)

            def clear_cache(self):
                self.clears += 1

        model = PerfectModel()
        with patch(
            "datasets.load_dataset",
            return_value=self.FakeDataset(rows),
        ):
            result = run_babilong(
                model,
                ["qa1"],
                ["256k"],
                num_samples=10,
                row_start=90,
                row_end=100,
                log_fn=lambda _: None,
            )
        self.assertEqual(result["results"]["qa1"]["256k"], 100.0)
        self.assertEqual(result["row_ids"]["qa1"]["256k"], list(range(90, 100)))
        self.assertEqual(model.clears, 1)

    def test_eval_inherits_pinned_revision_from_finetune_manifest(self):
        rows = [
            {"input": f"context {index}", "question": "Where?", "target": "kitchen"}
            for index in range(100)
        ]

        class PerfectModel:
            cfg = {"attn_type": "polar"}
            wip = []

            def __init__(self, checkpoint_dir):
                self.checkpoint_dir = checkpoint_dir

            def generate(self, prompts, max_tokens=16):
                return ["kitchen"] * len(prompts)

            def clear_cache(self):
                pass

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            manifest = {
                "dataset_id": "RMT-team/babilong",
                "dataset_revision": "immutable-dataset-sha",
                "prompt_protocol": "builtin-v1",
                "reserved_test_rows": [90, 100],
            }
            (Path(checkpoint_dir) / "finetune_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            model = PerfectModel(checkpoint_dir)
            with patch(
                "datasets.load_dataset",
                return_value=self.FakeDataset(rows),
            ) as loader:
                result = run_babilong(
                    model,
                    ["qa1"],
                    ["256k"],
                    num_samples=1,
                    row_start=90,
                    row_end=100,
                    log_fn=lambda _: None,
                )

        self.assertEqual(result["dataset_revision"], "immutable-dataset-sha")
        self.assertEqual(result["macro_average"]["256k"], 100.0)
        self.assertEqual(
            loader.call_args.kwargs["revision"],
            "immutable-dataset-sha",
        )


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

    def test_direct_generation_is_greedy_and_refuses_truncation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the lightweight development environment")

        class ToyModel:
            def __init__(self):
                self.embed = torch.nn.Embedding(4, 4)
                self.blocks = []
                self.norm = torch.nn.Identity()
                self.proj = torch.nn.Linear(4, 4, bias=False)
                with torch.no_grad():
                    self.embed.weight.copy_(torch.eye(4))
                    self.proj.weight.copy_(torch.eye(4))

        class ToyTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [int(piece) for piece in text.split()]

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(token) for token in ids if token != 0)

        scorer = object.__new__(DirectScorer)
        scorer.model = ToyModel()
        scorer.device = torch.device("cpu")
        scorer.max_length = 4
        scorer.tokenizer = ToyTokenizer()

        self.assertEqual(scorer.generate(["2"], max_tokens=2), ["2 2"])
        with self.assertRaisesRegex(ValueError, "silently truncated"):
            scorer.generate(["1 2 3"], max_tokens=2)

if __name__ == "__main__":
    unittest.main()
