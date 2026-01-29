#!/usr/bin/env python
"""Generate research questions from gap analysis report."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.gap_detection import GapDetectionConfig, load_gap_report  # noqa: E402
from src.analysis.research_questions import (  # noqa: E402
    QuestionScope,
    QuestionStyle,
    ResearchQuestionConfig,
    build_prompts_from_gap_report,
    format_questions_markdown,
    generate_questions_from_prompts,
)


def create_llm_caller(provider: str, model: str | None = None):
    """Create a callable that generates text from prompts.

    Args:
        provider: LLM provider ('anthropic', 'openai', 'google').
        model: Model identifier. If None, uses provider's default.

    Returns:
        Callable that takes a prompt and returns response text.
    """
    if provider == "anthropic":
        import anthropic
        from anthropic.types import TextBlock

        client = anthropic.Anthropic()
        model_name = model or "claude-sonnet-4-20250514"

        def call_anthropic(prompt: str) -> str:
            response = client.messages.create(
                model=model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            # Extract text from TextBlock
            for block in response.content:
                if isinstance(block, TextBlock):
                    return block.text
            return ""

        return call_anthropic

    elif provider == "openai":
        import openai

        client = openai.OpenAI()
        model_name = model or "gpt-4o"

        def call_openai(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content
            return content if content else ""

        return call_openai

    elif provider == "google":
        try:
            import google.generativeai as genai
        except ImportError as err:
            raise ImportError("google-generativeai package required for Google provider") from err

        model_name = model or "gemini-2.0-flash"
        gen_model = genai.GenerativeModel(model_name)

        def call_google(prompt: str) -> str:
            response = gen_model.generate_content(prompt)
            return response.text

        return call_google

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate research questions from gap analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate questions from existing gap report
  python scripts/research_questions.py --gap-report data/out/experiments/gap_analysis/gap_report.json

  # Run gap analysis first, then generate questions
  python scripts/research_questions.py --index-dir data/index --count 5

  # Use OpenAI instead of Anthropic
  python scripts/research_questions.py --gap-report report.json --provider openai

  # Generate narrow-scope causal questions
  python scripts/research_questions.py --gap-report report.json --scope narrow --styles causal comparative
""",
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--gap-report",
        type=Path,
        help="Path to existing gap analysis report (JSON)",
    )
    input_group.add_argument(
        "--index-dir",
        type=Path,
        help="Path to index directory (runs gap analysis first)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "out" / "experiments" / "research_questions",
        help="Directory to save outputs (default: data/out/experiments/research_questions)",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # LLM options
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google"],
        default="anthropic",
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (default: provider's default)",
    )

    # Generation options
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of questions per gap (default: 3)",
    )
    parser.add_argument(
        "--scope",
        choices=["narrow", "moderate", "broad"],
        default="moderate",
        help="Question scope (default: moderate)",
    )
    parser.add_argument(
        "--styles",
        nargs="+",
        choices=["exploratory", "causal", "comparative", "evaluative", "descriptive"],
        help="Allowed question styles (default: all)",
    )
    parser.add_argument(
        "--no-rationale",
        action="store_true",
        help="Disable rationale generation",
    )

    # Gap analysis options (when using --index-dir)
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Filter to specific collections (for gap analysis)",
    )
    parser.add_argument(
        "--max-gaps",
        type=int,
        default=5,
        help="Maximum gaps per category to process (default: 5)",
    )

    # Output control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts without calling LLM",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )

    return parser.parse_args()


def load_or_generate_gap_report(args: argparse.Namespace) -> dict:
    """Load or generate gap report based on arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        Gap report dictionary.
    """
    if args.gap_report:
        if not args.gap_report.exists():
            raise FileNotFoundError(f"Gap report not found: {args.gap_report}")
        with open(args.gap_report) as f:
            return json.load(f)
    else:
        # Run gap analysis
        if not args.index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {args.index_dir}")

        config = GapDetectionConfig(max_items=args.max_gaps)
        return load_gap_report(
            index_dir=args.index_dir,
            config=config,
            collections=args.collections,
        )


def main() -> int:
    """Generate research questions from gap analysis."""
    args = parse_args()

    # Validate output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate gap report
    if args.verbose:
        print("Loading gap report...")
    try:
        gap_report = load_or_generate_gap_report(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Build configuration
    scope_map = {
        "narrow": QuestionScope.NARROW,
        "moderate": QuestionScope.MODERATE,
        "broad": QuestionScope.BROAD,
    }
    style_map = {
        "exploratory": QuestionStyle.EXPLORATORY,
        "causal": QuestionStyle.CAUSAL,
        "comparative": QuestionStyle.COMPARATIVE,
        "evaluative": QuestionStyle.EVALUATIVE,
        "descriptive": QuestionStyle.DESCRIPTIVE,
    }

    styles = [style_map[s] for s in args.styles] if args.styles else []

    config = ResearchQuestionConfig(
        count=args.count,
        scope=scope_map[args.scope],
        styles=styles,
        include_rationale=not args.no_rationale,
    )

    # Build prompts
    if args.verbose:
        print("Building prompts from gap report...")
    prompts = build_prompts_from_gap_report(gap_report, config)

    if not prompts:
        print("No gaps found in report. Nothing to generate.")
        return 0

    if args.verbose:
        print(f"Generated {len(prompts)} prompts for gap types:")
        for p in prompts:
            gap_label = p["gap"].get("label") or p["gap"].get("direction") or "year gap"
            print(f"  - {p['type']}: {gap_label}")

    # Dry run: show prompts and exit
    if args.dry_run:
        print("\n=== DRY RUN: Prompts ===\n")
        for i, p in enumerate(prompts, 1):
            print(f"--- Prompt {i} ({p['type']}) ---")
            print(p["prompt"][:500] + "..." if len(p["prompt"]) > 500 else p["prompt"])
            print()
        return 0

    # Create LLM caller
    if args.verbose:
        print(f"Using provider: {args.provider}" + (f" ({args.model})" if args.model else ""))
    try:
        llm_caller = create_llm_caller(args.provider, args.model)
    except ImportError as e:
        print(f"Error: Missing dependency for provider '{args.provider}': {e}")
        return 1

    # Generate questions
    if args.verbose:
        print("Generating research questions...")

    result = generate_questions_from_prompts(prompts, llm_caller, config)

    if args.verbose:
        print(f"Generated {result.total_generated} questions")
        print(f"Removed {result.duplicates_removed} duplicates")
        print(f"Final questions: {len(result.questions)}")
        if result.generation_errors:
            print(f"Errors: {len(result.generation_errors)}")
            for err in result.generation_errors:
                print(f"  - {err}")

    # Save output
    if args.output_format == "markdown":
        output_content = format_questions_markdown(result)
        output_file = args.output_dir / "research_questions.md"
        with open(output_file, "w") as f:
            f.write(output_content)
    else:
        output_data = {
            "total_generated": result.total_generated,
            "duplicates_removed": result.duplicates_removed,
            "generation_errors": result.generation_errors,
            "questions": [
                {
                    "question": q.question,
                    "style": q.style,
                    "gap_type": q.gap_type,
                    "gap_label": q.gap_label,
                    "rationale": q.rationale,
                    "methodology_hints": q.methodology_hints,
                    "relevance_score": q.relevance_score,
                    "diversity_score": q.diversity_score,
                    "combined_score": q.combined_score,
                }
                for q in result.questions
            ],
        }
        output_file = args.output_dir / "research_questions.json"
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    print(f"Research questions saved to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
