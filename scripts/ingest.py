\"\"\"Load and chunk local PDFs for an initial ingestion smoke test.\"\"\"

import argparse
from pathlib import Path

from advanced_rag.ingestion.chunking import split_documents
from advanced_rag.ingestion.loaders import load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        default="data/raw/sample.pdf",
        help="Path to a PDF file (default: data/raw/sample.pdf)",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    documents = load_pdf(Path(args.pdf))
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    print(f"Loaded {len(documents)} pages and produced {len(chunks)} chunks.")
    if chunks:
        print(f"First chunk preview: {chunks[0].page_content[:300]!r}")


if __name__ == "__main__":
    main()
