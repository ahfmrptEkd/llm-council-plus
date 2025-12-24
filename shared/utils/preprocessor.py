"""Data Preprocessor for text classification based on keywords."""

from typing import Dict, List, Literal


class TextPreprocessor:
    """Preprocess text data and classify based on keywords."""

    # Keywords for classification
    BAD_KEYWORDS = ["bad", "argue", "conflict"]
    GOOD_KEYWORDS = ["charm", "love", "impress"]

    def __init__(self, bad_keywords: List[str] = None, good_keywords: List[str] = None, case_sensitive: bool = False):
        """Initialize the preprocessor.

        Args:
            bad_keywords: List of keywords that indicate "Bad" classification
            good_keywords: List of keywords that indicate "Good" classification
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.bad_keywords = bad_keywords or self.BAD_KEYWORDS
        self.good_keywords = good_keywords or self.GOOD_KEYWORDS
        self.case_sensitive = case_sensitive

        if not case_sensitive:
            self.bad_keywords = [kw.lower() for kw in self.bad_keywords]
            self.good_keywords = [kw.lower() for kw in self.good_keywords]

    def classify_text(self, text: str) -> Literal["Good", "Bad", "Neutral"]:
        """Classify text based on keywords.

        Args:
            text: Text to classify

        Returns:
            Classification result: "Good", "Bad", or "Neutral"
        """
        if not text:
            return "Neutral"

        search_text = text if self.case_sensitive else text.lower()

        # Check for bad keywords first
        has_bad = any(keyword in search_text for keyword in self.bad_keywords)
        if has_bad:
            return "Bad"

        # Check for good keywords
        has_good = any(keyword in search_text for keyword in self.good_keywords)
        if has_good:
            return "Good"

        return "Neutral"

    def analyze_text(self, text: str) -> Dict:
        """Analyze text and return detailed classification information.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with classification and matched keywords
        """
        if not text:
            return {"classification": "Neutral", "matched_keywords": [], "bad_matches": [], "good_matches": []}

        search_text = text if self.case_sensitive else text.lower()

        # Find matching keywords
        bad_matches = [kw for kw in self.bad_keywords if kw in search_text]
        good_matches = [kw for kw in self.good_keywords if kw in search_text]

        # Determine classification
        if bad_matches:
            classification = "Bad"
        elif good_matches:
            classification = "Good"
        else:
            classification = "Neutral"

        return {
            "classification": classification,
            "matched_keywords": bad_matches + good_matches,
            "bad_matches": bad_matches,
            "good_matches": good_matches,
            "text_length": len(text),
            "word_count": len(text.split()),
        }

    def process_batch(self, texts: List[str]) -> List[Dict]:
        """Process multiple texts.

        Args:
            texts: List of texts to process

        Returns:
            List of analysis results
        """
        return [self.analyze_text(text) for text in texts]

    def get_classification_stats(self, texts: List[str]) -> Dict:
        """Get statistics on classification results.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with classification statistics
        """
        results = self.process_batch(texts)

        stats = {
            "total": len(texts),
            "good": sum(1 for r in results if r["classification"] == "Good"),
            "bad": sum(1 for r in results if r["classification"] == "Bad"),
            "neutral": sum(1 for r in results if r["classification"] == "Neutral"),
        }

        stats["good_percentage"] = (stats["good"] / stats["total"] * 100) if stats["total"] > 0 else 0
        stats["bad_percentage"] = (stats["bad"] / stats["total"] * 100) if stats["total"] > 0 else 0
        stats["neutral_percentage"] = (stats["neutral"] / stats["total"] * 100) if stats["total"] > 0 else 0

        return stats

    def filter_by_classification(self, texts: List[str], classification: Literal["Good", "Bad", "Neutral"]) -> List[str]:
        """Filter texts by classification.

        Args:
            texts: List of texts to filter
            classification: Classification to filter by

        Returns:
            Filtered list of texts
        """
        return [text for text in texts if self.classify_text(text) == classification]
