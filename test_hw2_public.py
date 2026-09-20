"""Public smoke tests for CS 577 Homework 2.

Run from the HW2 directory with either command:

    python test_hw2_public.py
    python -m unittest -v test_hw2_public.py

Passing these tests does not guarantee full credit. Hidden tests use additional
inputs, and grading also considers derivations, explanations, plots, code
quality, and the executed notebook.
"""

import unittest

import numpy as np

import hw2_mlp as h


ATOL = 1e-10
RTOL = 1e-9


def fixed_case():
    X = np.array(
        [[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8], [0.7, -1.2]],
        dtype=np.float64,
    )
    y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)
    params = {
        "W1": np.array([[0.5, -0.3, 0.8], [-0.7, 0.6, 0.2]], dtype=np.float64),
        "b1": np.array([0.1, -0.2, 0.3], dtype=np.float64),
        "W2": np.array([[0.4], [-0.5], [0.7]], dtype=np.float64),
        "b2": np.array([-0.1], dtype=np.float64),
    }
    return X, y, params


class InterfaceAndPrimitiveTests(unittest.TestCase):
    def test_01_required_names_exist(self):
        required = [
            "sigmoid_stable",
            "relu",
            "binary_cross_entropy_with_logits",
            "initialize_parameters",
            "mlp_forward",
            "mlp_loss_and_gradients",
            "finite_difference_gradients",
            "predict_proba",
            "predict_labels",
            "binary_accuracy",
            "train_mlp",
            "make_xor_data",
        ]
        self.assertEqual([name for name in required if not hasattr(h, name)], [])

    def test_02_stable_sigmoid_extreme_and_reference_values(self):
        z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
        result = h.sigmoid_stable(z)
        self.assertEqual(result.shape, z.shape)
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0.0) & (result <= 1.0)))
        expected_middle = 1.0 / (1.0 + np.exp(-z[1:4]))
        np.testing.assert_allclose(result[1:4], expected_middle, atol=ATOL, rtol=RTOL)
        self.assertLess(result[0], 1e-15)
        self.assertGreater(result[-1], 1.0 - 1e-15)

    def test_03_relu_and_bce_with_logits(self):
        z = np.array([[-2.0, 0.0, 3.0]])
        np.testing.assert_array_equal(h.relu(z), np.array([[0.0, 0.0, 3.0]]))

        logits = np.array([[-3.0], [0.0], [2.0], [1000.0], [-1000.0]])
        y = np.array([[0.0], [1.0], [1.0], [1.0], [0.0]])
        result = h.binary_cross_entropy_with_logits(logits, y)
        expected = np.mean(
            np.maximum(logits, 0.0)
            - logits * y
            + np.log1p(np.exp(-np.abs(logits)))
        )
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        self.assertAlmostEqual(result, float(expected), places=12)

        with self.assertRaises(ValueError):
            h.binary_cross_entropy_with_logits(logits.reshape(-1), y)


class ForwardBackwardTests(unittest.TestCase):
    def test_04_initialization_shapes_dtype_and_determinism(self):
        p1 = h.initialize_parameters(2, 5, seed=31)
        p2 = h.initialize_parameters(2, 5, seed=31)
        self.assertEqual(tuple(p1.keys()), h.PARAMETER_KEYS)
        expected_shapes = {"W1": (2, 5), "b1": (5,), "W2": (5, 1), "b2": (1,)}
        for key in h.PARAMETER_KEYS:
            self.assertEqual(p1[key].shape, expected_shapes[key])
            self.assertEqual(p1[key].dtype, np.float64)
            np.testing.assert_array_equal(p1[key], p2[key])

    def test_05_forward_values_cache_and_no_mutation(self):
        X, _, params = fixed_case()
        X_before = X.copy()
        params_before = {key: value.copy() for key, value in params.items()}

        logits, cache = h.mlp_forward(X, params)
        Z1 = X @ params["W1"] + params["b1"]
        A1 = np.maximum(Z1, 0.0)
        expected = A1 @ params["W2"] + params["b2"]

        self.assertEqual(logits.shape, (4, 1))
        np.testing.assert_allclose(logits, expected, atol=ATOL, rtol=RTOL)
        for key in ["X", "Z1", "A1", "logits"]:
            self.assertIn(key, cache)
        np.testing.assert_array_equal(X, X_before)
        for key in h.PARAMETER_KEYS:
            np.testing.assert_array_equal(params[key], params_before[key])

    def test_06_manual_backward_values_shapes_and_no_update(self):
        X, y, params = fixed_case()
        params_before = {key: value.copy() for key, value in params.items()}
        loss, grads = h.mlp_loss_and_gradients(X, y, params)

        Z1 = X @ params["W1"] + params["b1"]
        A1 = np.maximum(Z1, 0.0)
        logits = A1 @ params["W2"] + params["b2"]
        expected_loss = np.mean(
            np.maximum(logits, 0.0)
            - y * logits
            + np.log1p(np.exp(-np.abs(logits)))
        )
        sigmoid = 1.0 / (1.0 + np.exp(-logits))
        dS = (sigmoid - y) / X.shape[0]
        expected = {
            "W2": A1.T @ dS,
            "b2": np.sum(dS, axis=0),
        }
        dA1 = dS @ params["W2"].T
        dZ1 = dA1 * (Z1 > 0.0)
        expected["W1"] = X.T @ dZ1
        expected["b1"] = np.sum(dZ1, axis=0)

        self.assertIsInstance(loss, float)
        self.assertAlmostEqual(loss, float(expected_loss), places=12)
        self.assertEqual(set(grads), set(h.PARAMETER_KEYS))
        for key in h.PARAMETER_KEYS:
            self.assertEqual(grads[key].shape, params[key].shape)
            np.testing.assert_allclose(grads[key], expected[key], atol=1e-9, rtol=1e-8)
            np.testing.assert_array_equal(params[key], params_before[key])

    def test_07_finite_differences_match_manual_and_do_not_mutate(self):
        X, y, params = fixed_case()
        before = {key: value.copy() for key, value in params.items()}
        _, manual = h.mlp_loss_and_gradients(X, y, params)
        numerical = h.finite_difference_gradients(X, y, params, epsilon=1e-5)
        self.assertEqual(set(numerical), set(h.PARAMETER_KEYS))
        for key in h.PARAMETER_KEYS:
            self.assertEqual(numerical[key].shape, params[key].shape)
            np.testing.assert_allclose(
                numerical[key],
                manual[key],
                atol=2e-6,
                rtol=2e-6,
                err_msg=f"finite-difference mismatch for {key}",
            )
            np.testing.assert_array_equal(params[key], before[key])


class ExperimentHelperTests(unittest.TestCase):
    def test_08_prediction_helpers_and_accuracy_shape_guard(self):
        X, y, params = fixed_case()
        probabilities = h.predict_proba(X, params)
        labels = h.predict_labels(X, params)
        self.assertEqual(probabilities.shape, y.shape)
        self.assertEqual(labels.shape, y.shape)
        self.assertTrue(np.all((probabilities >= 0.0) & (probabilities <= 1.0)))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        self.assertTrue(0.0 <= h.binary_accuracy(labels, y) <= 1.0)
        with self.assertRaises(ValueError):
            h.binary_accuracy(labels.reshape(-1), y)

    def test_09_xor_data_shape_balance_and_reproducibility(self):
        X1, y1 = h.make_xor_data(n_per_quadrant=10, noise=0.2, seed=91)
        X2, y2 = h.make_xor_data(n_per_quadrant=10, noise=0.2, seed=91)
        self.assertEqual(X1.shape, (40, 2))
        self.assertEqual(y1.shape, (40, 1))
        self.assertEqual(X1.dtype, np.float64)
        self.assertEqual(y1.dtype, np.float64)
        self.assertEqual(float(y1.mean()), 0.5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_10_training_returns_history_updates_and_learns(self):
        X, y = h.make_xor_data(n_per_quadrant=20, noise=0.22, seed=577)
        params, history = h.train_mlp(
            X, y, hidden_dim=8, steps=400, lr=0.2, seed=577
        )
        self.assertEqual(len(history), 400)
        self.assertTrue(np.all(np.isfinite(np.asarray(history))))
        self.assertLess(history[-1], history[0])
        self.assertEqual(set(params), set(h.PARAMETER_KEYS))
        accuracy = h.binary_accuracy(h.predict_labels(X, params), y)
        self.assertGreaterEqual(accuracy, 0.85)


if __name__ == "__main__":
    print("CS 577 HW2 public tests")
    print("Passing these tests does not guarantee full credit.\n")
    unittest.main(verbosity=2)
