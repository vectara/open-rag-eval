"""Main orchestration script for query generation."""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import OmegaConf

from open_rag_eval.query_generation import (
    LLMQueryGenerator,
    VectaraCorpusSource,
    LocalFileSource,
    CSVSource,
    OutputFormatter,
)
from open_rag_eval.models import (
    OpenAIModel,
    GeminiModel,
    AnthropicModel,
    TogetherModel,
)

logger = logging.getLogger(__name__)


def get_model(model_config: Dict):
    """
    Instantiate an LLM model from configuration.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Instantiated model object

    Raises:
        ValueError: If model type is invalid or configuration is missing
    """
    model_type = model_config.get("type")
    if not model_type:
        raise ValueError("Model type is required in configuration")

    model_classes = {
        "OpenAIModel": OpenAIModel,
        "GeminiModel": GeminiModel,
        "AnthropicModel": AnthropicModel,
        "TogetherModel": TogetherModel,
    }

    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(
            f"Invalid model type: {model_type}. "
            f"Must be one of: {list(model_classes.keys())}"
        )

    # Extract model options (name, api_key, etc.)
    model_options = {
        "name": model_config.get("name"),
        "api_key": model_config.get("api_key"),
    }

    # Add optional parameters
    if "base_url" in model_config:
        model_options["base_url"] = model_config["base_url"]

    return model_class(model_options)


def get_document_source(source_config: Dict):
    """
    Instantiate a document source from configuration.

    Args:
        source_config: Document source configuration dictionary

    Returns:
        Instantiated document source object

    Raises:
        ValueError: If source type is invalid or configuration is missing
    """
    source_type = source_config.get("type")
    if not source_type:
        raise ValueError("Document source type is required in configuration")

    options = source_config.get("options", {})

    if source_type == "VectaraCorpusSource":
        return VectaraCorpusSource(
            api_key=options.get("api_key"),
            corpus_key=options.get("corpus_key"),
        )

    if source_type == "LocalFileSource":
        return LocalFileSource(
            path=options.get("path"),
            file_extensions=options.get("file_extensions"),
        )

    if source_type == "CSVSource":
        return CSVSource(
            csv_path=options.get("csv_path"),
            text_column=options.get("text_column", "text"),
        )

    raise ValueError(
        f"Invalid document source type: {source_type}. "
        "Must be one of: VectaraCorpusSource, LocalFileSource, CSVSource"
    )


def generate_query_variant(
    generator: LLMQueryGenerator,
    documents: List[str],
    variant_config: Dict,
    output_config: Dict,
    base_filename: str,
    generate_expected_answers: bool = False
) -> None:
    """
    Generate a single query variant and save to file.

    Args:
        generator: Query generator instance
        documents: List of document texts
        variant_config: Variant configuration (name, min_words, max_words, n_questions)
        output_config: Output configuration
        base_filename: Base filename for output
        generate_expected_answers: Whether to generate expected answers with queries

    Raises:
        Exception: If generation or saving fails
    """
    variant_name = variant_config.get("name", "queries")
    n_questions = variant_config.get("n_questions", 50)
    min_words = variant_config.get("min_words", 5)
    max_words = variant_config.get("max_words", 20)

    logger.info("Generating variant: %s", variant_name)

    # Prepare output filename
    output_format = output_config.get("format", "csv")
    if variant_name and variant_name != "queries":
        output_filename = f"{base_filename}_{variant_name}.{output_format}"
    else:
        output_filename = f"{base_filename}.{output_format}"

    if generate_expected_answers:
        # Generate QA pairs with expected answers
        qa_pairs = generator.generate_with_answers(
            documents=documents,
            n_questions=n_questions,
            min_words=min_words,
            max_words=max_words,
        )

        OutputFormatter.save_qa_pairs(
            qa_pairs=qa_pairs,
            output_path=output_filename,
            output_format=output_format,
            include_metadata=output_config.get("include_metadata", False),
        )

        logger.info(
            "Saved %d QA pairs for variant '%s' to %s",
            len(qa_pairs),
            variant_name,
            output_filename
        )
    else:
        # Generate queries only
        queries = generator.generate(
            documents=documents,
            n_questions=n_questions,
            min_words=min_words,
            max_words=max_words,
        )

        OutputFormatter.save_queries(
            queries=queries,
            output_path=output_filename,
            output_format=output_format,
            include_metadata=output_config.get("include_metadata", False),
        )

        logger.info(
            "Saved %d queries for variant '%s' to %s",
            len(queries),
            variant_name,
            output_filename
        )


def run_query_generation(
    config_path: str,
    output_file: Optional[str] = None,
    num_queries: Optional[int] = None,
    dry_run: bool = False
) -> None:
    """
    Main orchestration function for query generation.

    Args:
        config_path: Path to configuration YAML file
        output_file: Optional output file path (overrides config)
        num_queries: Optional number of queries (overrides config)
        dry_run: If True, only show configuration without generating

    Raises:
        ValueError: If configuration is invalid
        IOError: If files cannot be read/written
    """
    # Load configuration
    logger.info("Loading configuration from %s", config_path)
    config = OmegaConf.load(config_path)

    # Initialize document source
    logger.info("Initializing document source...")
    document_source = get_document_source(config.document_source)

    # Load documents
    source_options = config.document_source.get("options", {})
    logger.info(
        "Loading documents (min_doc_size=%d, max_num_docs=%s)",
        source_options.get("min_doc_size", 0),
        source_options.get("max_num_docs", None)
    )
    documents = document_source.fetch_random_documents(
        min_doc_size=source_options.get("min_doc_size", 0),
        max_num_docs=source_options.get("max_num_docs", None),
        seed=source_options.get("seed", None)
    )

    if not documents:
        raise ValueError("No documents loaded. Check your document source configuration.")

    logger.info("Loaded %d documents", len(documents))

    if dry_run:
        logger.info("DRY RUN MODE: Configuration validated successfully")
        logger.info("Would generate queries from %d documents", len(documents))
        return

    # Initialize LLM model
    logger.info("Initializing LLM model...")
    model = get_model(config.model)

    # Initialize query generator
    # Convert OmegaConf to dict if present
    question_type_weights = config.generation.get("question_types", None)
    if question_type_weights is not None:
        question_type_weights = dict(question_type_weights)

    generator = LLMQueryGenerator(
        model=model,
        questions_per_doc=config.generation.get("questions_per_doc", 10),
        language=config.generation.get("language", "English"),
        question_type_weights=question_type_weights
    )

    # Determine base output filename
    output_config = config.get("output", {})
    base_filename = output_config.get("base_filename", "queries")

    if output_file:
        # Override with CLI argument
        base_filename = str(Path(output_file).stem)

    # Check if we should generate expected answers
    generate_expected_answers = config.generation.get("generate_expected_answers", False)
    if generate_expected_answers:
        logger.info("Will generate expected answers along with queries")

    # Generate queries based on variants or single generation
    variants = config.generation.get("variants", None)

    if variants:
        # Generate multiple variants
        logger.info("Generating %d query variants", len(variants))
        for variant in variants:
            generate_query_variant(
                generator=generator,
                documents=documents,
                variant_config=variant,
                output_config=output_config,
                base_filename=base_filename,
                generate_expected_answers=generate_expected_answers
            )
    else:
        # Single generation
        n_questions = num_queries or config.generation.get("n_questions", 50)
        min_words = config.generation.get("min_words", 5)
        max_words = config.generation.get("max_words", 20)
        output_format = output_config.get("format", "csv")
        output_filename = output_file or f"{base_filename}.{output_format}"

        if generate_expected_answers:
            logger.info("Generating %d QA pairs", n_questions)

            qa_pairs = generator.generate_with_answers(
                documents=documents,
                n_questions=n_questions,
                min_words=min_words,
                max_words=max_words,
            )

            OutputFormatter.save_qa_pairs(
                qa_pairs=qa_pairs,
                output_path=output_filename,
                output_format=output_format,
                include_metadata=output_config.get("include_metadata", False),
            )

            logger.info("Saved %d QA pairs to %s", len(qa_pairs), output_filename)
        else:
            logger.info("Generating %d queries", n_questions)

            queries = generator.generate(
                documents=documents,
                n_questions=n_questions,
                min_words=min_words,
                max_words=max_words,
            )

            OutputFormatter.save_queries(
                queries=queries,
                output_path=output_filename,
                output_format=output_format,
                include_metadata=output_config.get("include_metadata", False),
            )

            logger.info("Saved %d queries to %s", len(queries), output_filename)

    logger.info("Query generation complete!")


def main():
    """CLI entry point for standalone execution.

    This function maintains backwards compatibility by redirecting to the main CLI.
    It prepends 'generate-queries' to sys.argv to invoke the correct subcommand.
    """
    # Redirect to the main CLI with the generate-queries subcommand
    sys.argv.insert(1, 'generate-queries')
    from open_rag_eval.cli import main as cli_main  # pylint: disable=import-outside-toplevel,cyclic-import
    cli_main()


if __name__ == "__main__":
    main()
