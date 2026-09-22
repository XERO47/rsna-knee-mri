import unittest

from .. import auth


class AuthTests(unittest.TestCase):
    def test_tokens_are_unique_and_high_entropy(self):
        tokens = {auth.generate_token() for _ in range(50)}
        self.assertEqual(len(tokens), 50)
        self.assertTrue(all(len(token) >= 32 for token in tokens))

    def test_hash_is_deterministic_and_not_reversible_lookalike(self):
        token = auth.generate_token()
        self.assertEqual(auth.hash_token(token), auth.hash_token(token))
        self.assertNotEqual(auth.hash_token(token), token)

    def test_different_tokens_hash_differently(self):
        a, b = auth.generate_token(), auth.generate_token()
        self.assertNotEqual(auth.hash_token(a), auth.hash_token(b))

    def test_tokens_equal_constant_time_helper(self):
        token = auth.generate_token()
        self.assertTrue(auth.tokens_equal(token, token))
        self.assertFalse(auth.tokens_equal(token, auth.generate_token()))


if __name__ == "__main__":
    unittest.main()
