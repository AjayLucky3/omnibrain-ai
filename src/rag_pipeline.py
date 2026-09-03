from dataclasses import dataclass
from typing import Optional
import re

import ollama

from src.context_builder import ContextBuilder, RetrievedContext
from src.query_analyzer import (
    QueryAnalyzer,
    QueryAnalysis,
    DIRECT_LOOKUP,
    CALCULATION,
    COMPARISON,
    EXPLANATION,
    SUMMARY,
    OUT_OF_SCOPE,
)


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
    analysis: Optional[QueryAnalysis] = None


# ============================================================
# OLLAMA LLM
# ============================================================

class OllamaLLM:
    """
    Local LLM wrapper using Ollama.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
    ):

        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:
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
11. Do not mention retrieval, embeddings, chunks, or internal pipeline details.

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

    # ========================================================
    # VALUE EXTRACTION
    # ========================================================

    @staticmethod
    def extract_year_values(
        contexts: list[RetrievedContext],
        metric: str,
    ) -> dict[int, float]:

        combined_text = "\n".join(
            context.text
            for context in contexts
        )

        values: dict[int, float] = {}

        # ----------------------------------------------------
        # NORMALIZE TEXT
        # ----------------------------------------------------

        text = re.sub(
            r"(?i)operatingincome",
            "operating income",
            combined_text,
        )

        text = re.sub(
            r"(?i)revenueand",
            "revenue and",
            text,
        )

        metric = metric.lower().strip()

        # ----------------------------------------------------
        # TABLE EXTRACTION
        # ----------------------------------------------------

        table_rows = re.findall(
            r"(20\d{2})\s*\|\s*([\d,.]+)\s*\|\s*([\d,.]+)",
            text,
            flags=re.IGNORECASE,
        )

        if table_rows:

            for year, revenue, operating_income in table_rows:

                try:

                    if metric == "revenue":

                        values[int(year)] = float(
                            revenue.replace(",", "")
                        )

                    elif metric == "operating income":

                        values[int(year)] = float(
                            operating_income.replace(",", "")
                        )

                except (ValueError, TypeError):

                    continue

        # ----------------------------------------------------
        # SENTENCE EXTRACTION FALLBACK
        # ----------------------------------------------------

        if values:

            return values

        if metric == "operating income":

            pattern = (
                r"operating\s+income"
                r"\s+(?:was|is)"
                r"\s+([\d,.]+)"
                r"\s*(?:million)?"
                r"(?:\s+US\s+dollars?)?"
                r"\s+in\s+"
                r"(20\d{2})"
            )

            matches = re.findall(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            for value, year in matches:

                try:

                    values[int(year)] = float(
                        value.replace(",", "")
                    )

                except (ValueError, TypeError):

                    continue

        elif metric == "revenue":

            pattern = (
                r"(?:fiscal\s+year\s+)?"
                r"(20\d{2})"
                r".{0,100}?"
                r"revenue"
                r".{0,80}?"
                r"(?:of|to)"
                r"\s+"
                r"([\d,.]+)"
                r"\s*(?:million)?"
            )

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

                except (ValueError, TypeError):

                    continue

        return values

    # ========================================================
    # REGIONAL REVENUE
    # ========================================================

    @staticmethod
    def extract_regional_revenue(
        contexts: list[RetrievedContext],
    ) -> dict[str, float]:

        combined_text = "\n".join(
            context.text
            for context in contexts
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

    # ========================================================
    # NUMBER FORMATTING
    # ========================================================

    @staticmethod
    def format_number(
        value: float,
    ) -> str:

        if float(value).is_integer():

            return str(int(value))

        return f"{value:.2f}".rstrip("0").rstrip(".")

    # ========================================================
    # CALCULATE
    # ========================================================

    def calculate(
        self,
        question: str,
        analysis: QueryAnalysis,
        contexts: list[RetrievedContext],
    ) -> Optional[str]:
        """
        Perform deterministic calculations based on the
        QueryAnalyzer output.
        """

        q = analysis.normalized_query

        # ====================================================
        # REGIONAL COMPARISON
        # ====================================================

        if analysis.intent == COMPARISON:

            regional_values = (
                self.extract_regional_revenue(
                    contexts
                )
            )

            if not regional_values:

                return None

            # ------------------------------------------------
            # MOST / HIGHEST
            # ------------------------------------------------

            if (
                "most" in q
                or "highest" in q
            ):

                region, value = max(
                    regional_values.items(),
                    key=lambda item: item[1],
                )

                return (
                    f"{region} generated the most revenue, "
                    f"with {self.format_number(value)} "
                    f"million US dollars."
                )

            # ------------------------------------------------
            # LEAST / LOWEST
            # ------------------------------------------------

            if (
                "least" in q
                or "lowest" in q
            ):

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
        # DIFFERENCE BETWEEN TWO METRICS
        # ====================================================

        if (
            analysis.intent == CALCULATION
            and "difference between" in q
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

            if not revenue_values or not income_values:

                return None

            if not analysis.years:

                return None

            year = analysis.years[-1]

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
        # METRIC DETECTION
        # ====================================================

        metric = analysis.metric

        if not metric:

            return None

        values = self.extract_year_values(
            contexts,
            metric,
        )

        if not values:

            return None

        years = analysis.years

        # ====================================================
        # DIRECT LOOKUP
        # ====================================================

        if analysis.intent == DIRECT_LOOKUP:

            if not years:

                return None

            year = years[-1]

            if year not in values:

                return None

            value = values[year]

            return (
                f"{metric.capitalize()} in "
                f"{year} was "
                f"{self.format_number(value)} "
                f"million US dollars."
            )

        # ====================================================
        # CALCULATION BETWEEN TWO YEARS
        # ====================================================

        if (
            analysis.intent == CALCULATION
            and len(years) >= 2
        ):

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
            # ABSOLUTE DIFFERENCE
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

        return None


# ============================================================
# RAG PIPELINE
# ============================================================

class RAGPipeline:
    """
    Complete OmniBrain retrieval-augmented generation pipeline.

    Pipeline:

        User Query
             ↓
        Query Analyzer
             ↓
        Context Retrieval
             ↓
        Calculation Engine / LLM
             ↓
        Final Answer
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ):

        self.context_builder = ContextBuilder()

        self.query_analyzer = QueryAnalyzer()

        self.llm = OllamaLLM(
            model=model
        )

        self.calculation_engine = (
            CalculationEngine()
        )

        self.retrieval_limit = retrieval_limit

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    @staticmethod
    def build_prompt(
        question: str,
        contexts: list[RetrievedContext],
    ) -> str:

        context_text = "\n\n".join(
            (
                f"[Document: {context.document_id} | "
                f"Page: {context.page_number}]\n"
                f"{context.text}"
            )
            for context in contexts
        )

        return f"""
DOCUMENT CONTEXT:

{context_text}

USER QUESTION:

{question}

Answer the user's question using only the document context above.
"""

    # ========================================================
    # ASK
    # ========================================================

    def ask(
        self,
        question: str,
    ) -> RAGResult:

        # ====================================================
        # STEP 1 — QUERY ANALYSIS
        # ====================================================

        analysis = self.query_analyzer.analyze(
            question
        )

        # ====================================================
        # STEP 2 — OUT OF SCOPE
        # ====================================================

        if analysis.intent == OUT_OF_SCOPE:

            return RAGResult(
                answer=FALLBACK_ANSWER,
                contexts=[],
                analysis=analysis,
            )

        # ====================================================
        # STEP 3 — RETRIEVAL
        # ====================================================

        contexts = (
            self.context_builder.retrieve_context(
                query=analysis.normalized_query,
                limit=self.retrieval_limit,
            )
        )

        if not contexts:

            return RAGResult(
                answer=FALLBACK_ANSWER,
                contexts=[],
                analysis=analysis,
            )

        # ====================================================
        # STEP 4 — DETERMINISTIC CALCULATION
        # ====================================================

        calculated_answer = (
            self.calculation_engine.calculate(
                question=question,
                analysis=analysis,
                contexts=contexts,
            )
        )

        if calculated_answer:

            return RAGResult(
                answer=calculated_answer,
                contexts=contexts,
                analysis=analysis,
            )

        # ====================================================
        # STEP 5 — LLM FALLBACK
        # ====================================================

        prompt = self.build_prompt(
            question=analysis.normalized_query,
            contexts=contexts,
        )

        answer = self.llm.generate(
            prompt
        )

        # ====================================================
        # STEP 6 — RETURN RESULT
        # ====================================================

        return RAGResult(
            answer=answer,
            contexts=contexts,
            analysis=analysis,
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:

        self.context_builder.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    pipeline = RAGPipeline()

    try:

        print()
        print("=" * 60)
        print("OMNIBRAIN RAG PIPELINE")
        print("=" * 60)
        print()
        print("LLM: Ollama")
        print(f"Model: {OLLAMA_MODEL}")
        print()

        while True:

            question = input(
                "Enter your question: "
            ).strip()

            if not question:

                continue

            if question.lower() in {
                "exit",
                "quit",
            }:

                break

            result = pipeline.ask(
                question
            )

            print()
            print("=" * 60)
            print("QUERY ANALYSIS")
            print("=" * 60)

            if result.analysis:

                print(
                    f"Normalized: "
                    f"{result.analysis.normalized_query}"
                )

                print(
                    f"Intent: "
                    f"{result.analysis.intent}"
                )

                print(
                    f"Metric: "
                    f"{result.analysis.metric}"
                )

                print(
                    f"Years: "
                    f"{result.analysis.years}"
                )

                print(
                    f"Confidence: "
                    f"{result.analysis.confidence}"
                )

            print()
            print("=" * 60)
            print("QUESTION")
            print("=" * 60)
            print(question)

            print()
            print("=" * 60)
            print("ANSWER")
            print("=" * 60)
            print(result.answer)

            print()
            print("=" * 60)
            print("RETRIEVED SOURCES")
            print("=" * 60)

            for index, context in enumerate(
                result.contexts,
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

            print()

    finally:

        pipeline.close()