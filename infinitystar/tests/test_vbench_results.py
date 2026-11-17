"""Synthetic CLI fixtures; no VBench, video or model dependencies."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/parse_vbench2_eval_results.py'
spec = importlib.util.spec_from_file_location('scores', SCRIPT)
scores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scores)


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, value=1.0, spaces=False, scalar=False):
        for i, task in enumerate(scores.TASK_INFO):
            folder = self.root / task.replace(' ', '_')
            folder.mkdir(exist_ok=True)
            key = task if spaces else task.replace(' ', '_')
            number = value(i) if callable(value) else value
            data = number if scalar else [number, []]
            (folder / 'test_eval_results.json').write_text(json.dumps({key: data}), encoding='utf-8')

    def result_file(self):
        return self.root / 'Camera_Motion/test_eval_results.json'

    def run_cli(self, *options):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.root)] + list(options),
                              capture_output=True, text=True)

    def assert_invalid(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Camera Motion', result.stderr)
        self.assertNotIn('total score|', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_complete_controls(self):
        self.assertEqual(len(scores.TASK_INFO), 18)
        for value in [0.0, 1.0, 0.25]:
            with self.subTest(value=value):
                self.fixture(value)
                result = self.run_cli()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('total score|{:.4f}'.format(value), result.stdout)

    def test_nonuniform_scores(self):
        self.fixture(lambda i: i / 20)
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = sum(sum(scores.TASK_INFO.index(k) / 20 for k in group) / len(group)
                       for group in [scores.CREATIVITY_LIST, scores.COMMONSENSE_LIST,
                                     scores.CONTROLLABILITY_LIST, scores.HUMAN_FIDELITY_LIST,
                                     scores.PHYSICS_LIST]) / 5
        self.assertIn('total score|{:.4f}'.format(expected), result.stdout)
        self.assertIn('flat mean (19d)|0.4250', result.stdout)

    def test_empty_root(self):
        self.assert_invalid(self.run_cli())

    def test_missing_directory(self):
        self.fixture()
        self.result_file().unlink()
        self.result_file().parent.rmdir()
        self.assert_invalid(self.run_cli())

    def test_missing_file(self):
        self.fixture()
        self.result_file().unlink()
        self.assert_invalid(self.run_cli())

    def test_missing_key(self):
        self.fixture()
        self.result_file().write_text(json.dumps({'unrelated': 1.0}))
        self.assert_invalid(self.run_cli())

    def test_alternative_keys_and_value_forms(self):
        for spaces in [False, True]:
            for scalar in [False, True]:
                with self.subTest(spaces=spaces, scalar=scalar):
                    self.fixture(0.5, spaces=spaces, scalar=scalar)
                    result = self.run_cli()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('total score|0.5000', result.stdout)

    def test_malformed_data(self):
        self.fixture()
        for data in ['', '\n', '{', '{}', '[]', 'null', '{"Camera_Motion": []}',
                     '{"Camera_Motion": null}', '{"Camera_Motion": "bad"}',
                     '{"Camera_Motion": [null]}', '{"Camera_Motion": NaN}',
                     '{"Camera_Motion": Infinity}']:
            for options in [(), ('--allow-partial',)]:
                with self.subTest(data=data, options=options):
                    self.result_file().write_text(data)
                    self.assert_invalid(self.run_cli(*options))

    def test_explicit_partial(self):
        folder = self.root / 'Diversity'
        folder.mkdir()
        (folder / 'test_eval_results.json').write_text('{"Diversity": [0.0, []]}')
        result = self.run_cli('--allow-partial')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('1/18', result.stdout)
        self.assertIn('Diversity: 0.0000', result.stdout)
        self.assertNotIn('Camera Motion:', result.stdout)
        self.assertIn('Camera Motion', result.stderr)
        self.assertNotIn('total score|', result.stdout)
        self.assertNotIn('flat mean', result.stdout)
        self.assertNotIn('creativity score|', result.stdout)

    def test_allow_partial_with_complete_input(self):
        self.fixture()
        result = self.run_cli('--allow-partial')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('total score|1.0000', result.stdout)

    def test_underscore_key_precedence(self):
        self.fixture()
        self.result_file().write_text('{"Camera Motion": 0.0, "Camera_Motion": [1.0, []]}')
        result = self.run_cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('total score|1.0000', result.stdout)

    def test_partial_missing_file(self):
        self.fixture()
        self.result_file().unlink()
        result = self.run_cli('--allow-partial')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('17/18', result.stdout)
        self.assertIn('Camera Motion: missing result file', result.stderr)
        self.assertNotIn('total score|', result.stdout)

    def test_empty_explicit_partial(self):
        result = self.run_cli('--allow-partial')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('0/18', result.stdout)
        self.assertNotIn('total score|', result.stdout)


if __name__ == '__main__':
    unittest.main()
