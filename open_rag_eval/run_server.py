#!/usr/bin/env python3
import argparse
from open_rag_eval.api.server import run_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the open-rag-eval API server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )

    args = parser.parse_args()

    print(f"Starting open-rag-eval server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port, debug=args.debug)
