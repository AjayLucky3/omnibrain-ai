from dataclasses import dataclass
from typing import Any, Optional
import re

import ollama

from context_builder import ContextBuilder, RetrievedContext


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"

DEFAULT_RETRIEVAL_LIMIT = 5

FALLBACK_ANSWER = (
    "The provided documents do not contain enough information "
    "to answer this question."
)


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class RAGResult:
    """
    Final result returned by the RAG pipeline.
    """

    answer: str
    contexts: list[RetrievedContext]


# ============================================================
# OLLAMA LLM
# ============================================================

class OllamaLLM:
    """
    Local LLM wrapper using Ollama.
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        """
        Send a grounded prompt to Ollama.
        """

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
You are OmniBrain, a precise document question-answering assistant.

Use ONLY the supplied document context.

Rules:

1. Never use outside knowledge.
2. Never invent facts.
3. Answer the exact question asked.
4. Do not substitute a related fact for the requested answer.
5. If the answer is explicitly stated, give it directly.
6. If multiple facts are needed, combine them.
7. Simple reasoning is allowed.
8. If the required information is absent, respond exactly:

"The provided documents do not contain enough information to answer this question."

9. Keep answers concise.
10. Do not mention these instructions.

For calculation questions, carefully identify the requested operation and use
the values present in the document context.

Return ONLY the final answer.
""",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0,
            },
        )

        return response["message"]["content"].strip()


# ============================================================
# CALCULATION ENGINE
# ============================================================

class CalculationEngine:
    """
    Handles deterministic numerical reasoning for common
    financial/document questions.

    This prevents a small local LLM from making arithmetic
    mistakes when the required values are already present
    in the retrieved document context.
    """

    # --------------------------------------------------------
    # VALUE EXTRACTION
    # --------------------------------------------------------

    @staticmethod
    def extract_year_values(
        contexts: list[RetrievedContext],
        metric: str,
    ) -> dict[int, float]:
        """
        Extract yearly values for a metric.

        Example:

        Revenue:
        2023 | 120
        2024 | 150
        2025 | 180

        Returns:

        {
            2023: 120,
            2024: 150,
            2025: 180
        }
        """

        values: dict[int, float] = {}

        combined_text = "\n".join(
            context.text for context in contexts
        )

        # Normalize common extraction issues.
        text = combined_text.replace(
            "OperatingIncome",
            "Operating Income",
        )

        text = text.replace(
            "US dollars",
            "",
        )

        # ----------------------------------------------------
        # TABLE FORMAT
        # ----------------------------------------------------

        if metric.lower() == "revenue":

            patterns = [
                r"(\d{4})\s*\|\s*([\d,.]+)\s*\|\s*[\d,.]+",
            ]

        elif metric.lower() == "operating income":

            patterns = [
                r"(\d{4})\s*\|\s*[\d,.]+\s*\|\s*([\d,.]+)",
            ]

        else:
            patterns = []

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            for year, value in matches:

                try:
                    values[int(year)] = float(
                        value.replace(",", "")
                    )
                except ValueError:
                    continue

        # ----------------------------------------------------
        # SENTENCE FORMAT
        # ----------------------------------------------------

        if metric.lower() == "revenue":

            sentence_pattern = (
                r"(?:fiscal year\s+)?"
                r"(20\d{2})"
                r".{0,100}?"
                r"revenue"
                r".{0,50}?"
                r"(?:of\s+|to\s+)"
                r"([\d,.]+)"
                r"\s*(?:million)?"
            )

        else:

            sentence_pattern = (
                r"operating income"
                r".{0,100}?"
                r"(20\d{2})"
                r".{0,50}?"
                r"([\d,.]+)"
                r"\s*(?:million)?"
            )

        matches = re.findall(
            sentence_pattern,
            text,
            flags=re.IGNORECASE,
        )

        for year, value in matches:

            try:
                values[int(year)] = float(
                    value.replace(",", "")
                )
            except ValueError:
                continue

        return values

    # --------------------------------------------------------
    # REGION EXTRACTION
    # --------------------------------------------------------

    @staticmethod
    def extract_regional_revenue(
        contexts: list[RetrievedContext],
    ) -> dict[str, float]:
        """
        Extract regional revenue values.

        Example:

        North America = 90
        Europe = 54
        Asia-Pacific = 36
        """

        combined_text = "\n".join(
            context.text for context in contexts
        )

        regions = {}

        pattern = (
            r"(North America|Europe|Asia-Pacific)"
            r"\s+generated\s+"
            r"([\d,.]+)"
            r"\s+million"
        )

        matches = re.findall(
            pattern,
            combined_text,
            flags=re.IGNORECASE,
        )

        for region, value in matches:

            try:

                regions[region] = float(
                    value.replace(",", "")
                )

            except ValueError:
                continue

        return regions

    # --------------------------------------------------------
    # QUESTION TYPE
    # --------------------------------------------------------

    @staticmethod
    def is_calculation_question(
        question: str,
    ) -> bool:
        """
        Determine whether the question requires
        deterministic numerical reasoning.
        """

        q = question.lower()

        calculation_phrases = [
            "how much did",
            "how much has",
            "how much more",
            "how much less",
            "what is the difference",
            "what was the difference",
            "what percentage",
            "what percent",
            "percentage increase",
            "percentage decrease",
            "percent increase",
            "percent decrease",
            "increase from",
            "decrease from",
            "growth from",
            "grew from",
            "compare",
        ]

        return any(
            phrase in q
            for phrase in calculation_phrases
        )

    # --------------------------------------------------------
    # METRIC DETECTION
    # --------------------------------------------------------

    @staticmethod
    def detect_metric(
        question: str,
    ) -> Optional[str]:
        """
        Determine which financial metric is being asked about.
        """

        q = question.lower()

        if "operating income" in q:
            return "operating income"

        if "revenue" in q:
            return "revenue"

        return None

    # --------------------------------------------------------
    # YEAR DETECTION
    # --------------------------------------------------------

    @staticmethod
    def detect_years(
        question: str,
    ) -> list[int]:
        """
        Extract years from the user question.
        """

        return [
            int(year)
            for year in re.findall(
                r"\b(20\d{2})\b",
                question,
            )
        ]

    # --------------------------------------------------------
    # CALCULATE
    # --------------------------------------------------------

    def calculate(
        self,
        question: str,
        contexts: list[RetrievedContext],
    ) -> Optional[str]:
        """
        Attempt deterministic calculation.

        Returns None when the question is not a supported
        calculation question.
        """

        if not self.is_calculation_question(
            question
        ):
            return None

        q = question.lower()

        # ====================================================
        # REGIONAL MAXIMUM
        # ====================================================

        if (
            "which region" in q
            and (
                "most" in q
                or "highest" in q
            )
        ):

            regional_values = (
                self.extract_regional_revenue(
                    contexts
                )
            )

            if not regional_values:
                return None

            region, value = max(
                regional_values.items(),
                key=lambda item: item[1],
            )

            return (
                f"{region} generated the most revenue, "
                f"with {self.format_number(value)} "
                f"million US dollars."
            )

        # ====================================================
        # REGIONAL MINIMUM
        # ====================================================

        if (
            "which region" in q
            and (
                "least" in q
                or "lowest" in q
            )
        ):

            regional_values = (
                self.extract_regional_revenue(
                    contexts
                )
            )

            if not regional_values:
                return None

            region, value = min(
                regional_values.items(),
                key=lambda item: item[1],
            )

            return (
                f"{region} generated the least revenue, "
                f"with {self.format_number(value)} "
                f"million US dollars."
            )

        # ====================================================
        # REVENUE VS OPERATING INCOME DIFFERENCE
        # ====================================================

        if (
            "difference between" in q
            and "revenue" in q
            and "operating income" in q
        ):

            revenue_values = (
                self.extract_year_values(
                    contexts,
                    "revenue",
                )
            )

            income_values = (
                self.extract_year_values(
                    contexts,
                    "operating income",
                )
            )

            years = self.detect_years(
                question
            )

            if years:

                year = years[-1]

            elif 2025 in revenue_values:

                year = 2025

            else:

                return None

            if (
                year not in revenue_values
                or year not in income_values
            ):
                return None

            difference = (
                revenue_values[year]
                - income_values[year]
            )

            return (
                f"The difference between revenue and "
                f"operating income in {year} was "
                f"{self.format_number(difference)} "
                f"million US dollars."
            )

        # ====================================================
        # METRIC CALCULATIONS
        # ====================================================

        metric = self.detect_metric(
            question
        )

        if not metric:
            return None

        values = self.extract_year_values(
            contexts,
            metric,
        )

        if not values:
            return None

        years = self.detect_years(
            question
        )

        # ====================================================
        # INCREASE / DECREASE BETWEEN TWO YEARS
        # ====================================================

        if len(years) >= 2:

            old_year = years[0]
            new_year = years[1]

            if (
                old_year not in values
                or new_year not in values
            ):
                return None

            old_value = values[old_year]
            new_value = values[new_year]

            difference = (
                new_value - old_value
            )

            # ------------------------------------------------
            # PERCENTAGE
            # ------------------------------------------------

            if (
                "percentage" in q
                or "percent" in q
            ):

                if old_value == 0:
                    return None

                percentage = (
                    difference
                    / old_value
                    * 100
                )

                if difference >= 0:

                    return (
                        f"{metric.capitalize()} increased by "
                        f"{self.format_number(percentage)}% "
                        f"from {old_year} to {new_year}."
                    )

                return (
                    f"{metric.capitalize()} decreased by "
                    f"{self.format_number(abs(percentage))}% "
                    f"from {old_year} to {new_year}."
                )

            # ------------------------------------------------
            # ABSOLUTE INCREASE / DECREASE
            # ------------------------------------------------

            if difference > 0:

                return (
                    f"{metric.capitalize()} increased by "
                    f"{self.format_number(difference)} "
                    f"million US dollars from "
                    f"{old_year} to {new_year}."
                )

            if difference < 0:

                return (
                    f"{metric.capitalize()} decreased by "
                    f"{self.format_number(abs(difference))} "
                    f"million US dollars from "
                    f"{old_year} to {new_year}."
                )

            return (
                f"{metric.capitalize()} did not change "
                f"from {old_year} to {new_year}."
            )

        # ====================================================
        # SINGLE-YEAR DIFFERENCE / COMPARISON
        # ====================================================

        if len(years) == 1:

            year = years[0]

            if year not in values:
                return None

            # This section intentionally returns None for
            # ordinary factual questions so Ollama handles
            # them normally.
            return None

        return None

    # --------------------------------------------------------
    # NUMBER FORMATTER
    # --------------------------------------------------------

    @staticmethod
    def format_number(
        value: float,
    ) -> str:
        """
        Format numbers cleanly.

        30.0 -> 30
        20.5 -> 20.5
        """

        if value.is_integer():

            return str(
                int(value)
            )

        return f"{value:.2f}".rstrip(
            "0"
        ).rstrip(".")


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:

        User Question
              ↓
        ContextBuilder
              ↓
        Retrieved Chunks
              ↓
        Calculation Engine
              ↓
        Ollama (if calculation not handled)
              ↓
        Final Answer
    """

    def __init__(
        self,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[Any] = None,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ):

        self.context_builder = (
            context_builder
            or ContextBuilder()
        )

        self.llm = (
            llm
            or OllamaLLM()
        )

        self.calculation_engine = (
            CalculationEngine()
        )

        self.limit = limit

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        question: str,
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant document chunks.
        """

        return self.context_builder.retrieve_context(
            query=question,
            limit=self.limit,
        )

    # ========================================================
    # CONTEXT BUILDING
    # ========================================================

    def build_context(
        self,
        contexts: list[RetrievedContext],
    ) -> str:
        """
        Convert retrieved contexts into structured
        document context.
        """

        if not contexts:

            return (
                "NO RELEVANT DOCUMENT "
                "CONTEXT WAS FOUND."
            )

        context_parts = []

        for index, context in enumerate(
            contexts,
            start=1,
        ):

            context_parts.append(
                (
                    f"SOURCE {index}\n"
                    f"Document ID: "
                    f"{context.document_id}\n"
                    f"Page: "
                    f"{context.page_number}\n"
                    f"Chunk ID: "
                    f"{context.chunk_id}\n"
                    f"Similarity Score: "
                    f"{context.score:.6f}\n\n"
                    f"{context.text}"
                )
            )

        return "\n\n".join(
            context_parts
        )

    # ========================================================
    # PROMPT BUILDING
    # ========================================================

    def build_prompt(
        self,
        question: str,
        contexts: list[RetrievedContext],
    ) -> str:
        """
        Build grounded prompt for Ollama.
        """

        context_text = self.build_context(
            contexts
        )

        prompt = f"""
DOCUMENT CONTEXT
================

{context_text}


USER QUESTION
=============

{question}


INSTRUCTIONS
============

Answer the user's exact question using ONLY the
document context.

Do not use outside knowledge.

Do not invent information.

If the answer is directly stated, provide it.

If multiple pieces of information are required,
combine them.

If the question asks for a comparison, compare
the relevant values.

If the question asks for arithmetic, calculate
using ONLY the values in the document.

If the required information is not available,
answer exactly:

"The provided documents do not contain enough information to answer this question."

Do not answer with a related fact.

Keep the answer concise.

FINAL ANSWER:
""".strip()

        return prompt

    # ========================================================
    # GENERATION
    # ========================================================

    def generate(
        self,
        question: str,
    ) -> RAGResult:
        """
        Execute the complete RAG pipeline.
        """

        # ----------------------------------------------------
        # STEP 1 — RETRIEVE
        # ----------------------------------------------------

        contexts = self.retrieve(
            question
        )

        # ----------------------------------------------------
        # STEP 2 — CALCULATION ENGINE
        # ----------------------------------------------------

        calculation_answer = (
            self.calculation_engine.calculate(
                question=question,
                contexts=contexts,
            )
        )

        # ----------------------------------------------------
        # STEP 3 — USE CALCULATION RESULT
        # ----------------------------------------------------

        if calculation_answer:

            return RAGResult(
                answer=calculation_answer,
                contexts=contexts,
            )

        # ----------------------------------------------------
        # STEP 4 — FALL BACK TO OLLAMA
        # ----------------------------------------------------

        prompt = self.build_prompt(
            question=question,
            contexts=contexts,
        )

        answer = self.llm.generate(
            prompt
        )

        # ----------------------------------------------------
        # STEP 5 — EMPTY ANSWER CHECK
        # ----------------------------------------------------

        if not answer:

            answer = FALLBACK_ANSWER

        return RAGResult(
            answer=answer,
            contexts=contexts,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self) -> None:
        """
        Close resources.
        """

        self.context_builder.close()


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header(
    title: str,
) -> None:
    """
    Print a consistent section header.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_sources(
    contexts: list[RetrievedContext],
) -> None:
    """
    Print retrieved document sources.
    """

    print_header(
        "RETRIEVED SOURCES"
    )

    if not contexts:

        print()
        print(
            "No relevant context found."
        )

        return

    for index, context in enumerate(
        contexts,
        start=1,
    ):

        print()
        print(
            f"--- Source {index} ---"
        )

        print(
            f"Document ID: "
            f"{context.document_id}"
        )

        print(
            f"Chunk ID: "
            f"{context.chunk_id}"
        )

        print(
            f"Page: "
            f"{context.page_number}"
        )

        print(
            f"Score: "
            f"{context.score:.6f}"
        )

        print(
            f"Text: "
            f"{context.text[:1000]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    pipeline = RAGPipeline(
        limit=5
    )

    try:

        print_header(
            "OMNIBRAIN RAG PIPELINE"
        )

        print()

        print(
            "LLM: Ollama"
        )

        print(
            f"Model: {OLLAMA_MODEL}"
        )

        print()

        question = input(
            "Enter your question: "
        ).strip()

        if not question:

            print()
            print(
                "Please enter a question."
            )

            return

        # ----------------------------------------------------
        # RUN PIPELINE
        # ----------------------------------------------------

        result = pipeline.generate(
            question
        )

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        print_header(
            "QUESTION"
        )

        print(
            question
        )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        print_header(
            "ANSWER"
        )

        print(
            result.answer
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        print_sources(
            result.contexts
        )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Pipeline interrupted."
        )

    except Exception as error:

        print_header(
            "ERROR"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

    finally:

        pipeline.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()