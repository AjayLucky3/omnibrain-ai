from dataclasses import dataclass
import re


# ============================================================
# QUERY INTENTS
# ============================================================

DIRECT_LOOKUP = "direct_lookup"
CALCULATION = "calculation"
COMPARISON = "comparison"
EXPLANATION = "explanation"
SUMMARY = "summary"
OUT_OF_SCOPE = "out_of_scope"


# ============================================================
# QUERY ANALYSIS RESULT
# ============================================================

@dataclass
class QueryAnalysis:
    """
    Represents the interpreted user query.
    """

    original_query: str
    normalized_query: str
    intent: str
    metric: str | None
    metrics: list[str]
    years: list[int]
    confidence: float


# ============================================================
# QUERY ANALYZER
# ============================================================

class QueryAnalyzer:
    """
    Analyzes user questions before retrieval.

    Responsibilities:

    1. Normalize the question.
    2. Detect the user's intent.
    3. Detect financial metrics.
    4. Detect years.
    5. Provide a confidence score.
    """

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize(query: str) -> str:
        """
        Normalize common variations and simple typos.
        """

        query = query.lower().strip()

        # ----------------------------------------------------
        # Common spelling corrections
        # ----------------------------------------------------

        replacements = {
            "operatinfg": "operating",
            "operatng": "operating",
            "operatign": "operating",
            "operatng": "operating",
            "revenuee": "revenue",
            "revnue": "revenue",
            "reveneu": "revenue",
            "incom": "income",
            "diffrence": "difference",
            "differnce": "difference",
            "percantage": "percentage",
            "percentege": "percentage",
        }

        for wrong, correct in replacements.items():

            query = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                query,
            )

        # ----------------------------------------------------
        # Normalize whitespace
        # ----------------------------------------------------

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query

    # ========================================================
    # METRIC DETECTION
    # ========================================================

    @staticmethod
    def detect_metrics(
        query: str,
    ) -> list[str]:
        """
        Detect all financial metrics mentioned
        in the question.
        """

        metrics = []

        # ----------------------------------------------------
        # Operating Income
        # ----------------------------------------------------

        if "operating income" in query:

            metrics.append(
                "operating income"
            )

        # ----------------------------------------------------
        # Revenue
        # ----------------------------------------------------

        if "revenue" in query:

            metrics.append(
                "revenue"
            )

        return metrics

    # ========================================================
    # MAIN METRIC
    # ========================================================

    @staticmethod
    def detect_metric(
        query: str,
    ) -> str | None:
        """
        Detect the primary financial metric.

        The first detected metric is treated as
        the primary metric for backward compatibility.
        """

        metrics = QueryAnalyzer.detect_metrics(
            query
        )

        if metrics:

            return metrics[0]

        return None

    # ========================================================
    # YEAR DETECTION
    # ========================================================

    @staticmethod
    def detect_years(
        query: str,
    ) -> list[int]:
        """
        Extract years such as 2023, 2024, 2025.
        """

        return [
            int(year)
            for year in re.findall(
                r"\b20\d{2}\b",
                query,
            )
        ]

    # ========================================================
    # INTENT DETECTION
    # ========================================================

    @staticmethod
    def detect_intent(
        query: str,
    ) -> tuple[str, float]:
        """
        Determine the likely intent of the question.
        """

        # ----------------------------------------------------
        # OUT OF SCOPE
        # ----------------------------------------------------

        out_of_scope_patterns = [
            "capital of",
            "president of",
            "weather in",
            "population of",
            "who is",
            "where is",
        ]

        if any(
            pattern in query
            for pattern in out_of_scope_patterns
        ):

            return OUT_OF_SCOPE, 0.95

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        comparison_patterns = [
            "which region",
            "which company",
            "which year",
            "most",
            "highest",
            "least",
            "lowest",
            "compare",
        ]

        if any(
            pattern in query
            for pattern in comparison_patterns
        ):

            return COMPARISON, 0.90

        # ----------------------------------------------------
        # CALCULATION
        # ----------------------------------------------------

        calculation_patterns = [
            "how much did",
            "how much more",
            "how much less",
            "what is the difference",
            "what was the difference",
            "percentage increase",
            "percentage decrease",
            "percent increase",
            "percent decrease",
            "what percentage",
            "what percent",
            "increase from",
            "decrease from",
            "growth from",
            "grew from",
        ]

        if any(
            pattern in query
            for pattern in calculation_patterns
        ):

            return CALCULATION, 0.95

        # ----------------------------------------------------
        # EXPLANATION
        # ----------------------------------------------------

        explanation_patterns = [
            "why",
            "what caused",
            "what reason",
            "reason for",
            "explain",
            "how did",
        ]

        if any(
            pattern in query
            for pattern in explanation_patterns
        ):

            return EXPLANATION, 0.90

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_patterns = [
            "summarize",
            "summary",
            "overview",
            "what does the document say",
            "tell me about",
        ]

        if any(
            pattern in query
            for pattern in summary_patterns
        ):

            return SUMMARY, 0.85

        # ----------------------------------------------------
        # DIRECT LOOKUP
        # ----------------------------------------------------

        lookup_patterns = [
            "what was",
            "what is",
            "how much was",
            "how much is",
            "when was",
            "who was",
            "where was",
        ]

        if any(
            pattern in query
            for pattern in lookup_patterns
        ):

            return DIRECT_LOOKUP, 0.90

        # ----------------------------------------------------
        # DEFAULT
        # ----------------------------------------------------

        return DIRECT_LOOKUP, 0.50

    # ========================================================
    # ANALYZE
    # ========================================================

    def analyze(
        self,
        query: str,
    ) -> QueryAnalysis:
        """
        Analyze a complete user query.
        """

        normalized_query = self.normalize(
            query
        )

        intent, confidence = (
            self.detect_intent(
                normalized_query
            )
        )

        metrics = self.detect_metrics(
            normalized_query
        )

        metric = self.detect_metric(
            normalized_query
        )

        years = self.detect_years(
            normalized_query
        )

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            metric=metric,
            metrics=metrics,
            years=years,
            confidence=confidence,
        )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    analyzer = QueryAnalyzer()

    print()
    print("=" * 60)
    print("OMNIBRAIN QUERY ANALYZER")
    print("=" * 60)

    test_questions = [
        "What was revenue in 2024?",
        "What was operating income in 2025?",
        "how much did revenue increase from 2024 to 2025?",
        "what was the difference between revenue and operating income in 2025?",
        "Which region generated the most revenue in 2025?",
        "What reason did management give for increased operating expenses?",
        "Summarize the financial report.",
        "What was the capital of France?",
        "what was operatinfg income in 2025?",
    ]

    for question in test_questions:

        result = analyzer.analyze(
            question
        )

        print()
        print("-" * 60)

        print(
            f"Question: "
            f"{result.original_query}"
        )

        print(
            f"Normalized: "
            f"{result.normalized_query}"
        )

        print(
            f"Intent: "
            f"{result.intent}"
        )

        print(
            f"Metric: "
            f"{result.metric}"
        )

        print(
            f"Metrics: "
            f"{result.metrics}"
        )

        print(
            f"Years: "
            f"{result.years}"
        )

        print(
            f"Confidence: "
            f"{result.confidence}"
        )