import unittest
import sys
import os
sys.path.append(os.getcwd())

from citiTrends.trends import TopTrends


class TestTopTrends(unittest.TestCase):
    
    def test_get_freqs(self):
        stop_words = ['the', 'is', 'in']
        tt = TopTrends(stop_words=stop_words, bigrams=False, min_freq=0.01)
        texts = ["the sun is bright", "the sky is blue", "the sun is in the sky"]
        freqs = tt.get_freqs(texts)
        expected_freqs = {'blue': 0.3333333333333333, 'bright': 0.3333333333333333, 'sky': 0.6666666666666666, 'sun': 0.6666666666666666}
        self.assertEqual(freqs, expected_freqs)
        
    def test_is_trend(self):
        tt = TopTrends()
        self.assertTrue(tt.is_trend(0.05, 0.01, 2))
        self.assertFalse(tt.is_trend(0.02, 0.01, 2))
    
    def test_get_trends(self):
        tt = TopTrends(bigrams=False)
        texts_present = ["trend", "trend", "a"]
        texts_past = ["trend term", "top term", "a"]
        rate = 0.2
        trends = tt.get_trends(texts_present, texts_past, rate)
        expected_trends = {"trend": (2/3-1/3)/(1/3)}
        self.assertEqual(trends, expected_trends)


    def test_get_trends_high_rate(self):
        tt = TopTrends(bigrams=False)
        texts_present = ["trend", "trend", "a"]
        texts_past = ["trend term", "top term", "a"]
        rate = 1.5
        trends = tt.get_trends(texts_present, texts_past, rate)
        expected_trends = {}
        self.assertEqual(trends, expected_trends)

    def test_get_trends_new_term(self):
        tt = TopTrends(bigrams=False)
        texts_present = ["trend", "trend", "a"]
        texts_past = ["a term", "top term", "a"]
        rate = 1.5
        trends = tt.get_trends(texts_present, texts_past, rate)
        expected_trends = {"trend": 9999}
        self.assertEqual(trends, expected_trends)
    