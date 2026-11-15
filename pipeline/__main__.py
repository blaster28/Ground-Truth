"""Ligne de commande du pipeline (appelée par run.sh).

    python -m pipeline index    --pdf-dir corpus/pdf --index-dir index/
    python -m pipeline update   --index-dir index/ --pdf-dir corpus/pdf_v2
    python -m pipeline retrieve --index-dir index/ --questions q.json --out runs/hits.json
    python -m pipeline generate --retrieved runs/hits.json --out runs/pred.json --backend hf

Les options (--chunking, --mode, --no-rerank, --tau, --no-check...) servent aux ablations.
"""
import argparse

from . import answer, index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("index")
    a.add_argument("--pdf-dir", required=True)
    a.add_argument("--index-dir", required=True)
    a.add_argument("--chunking", default="structure", choices=["page", "fenetre", "structure"])
    a.add_argument("--no-context", action="store_true", help="sans préfixe titre/date sur les chunks")
    a.add_argument("--ocr", default="auto", choices=["auto", "lighton", "rapidocr", "none"])

    u = sub.add_parser("update")
    u.add_argument("--index-dir", required=True)
    u.add_argument("--pdf-dir", required=True)

    r = sub.add_parser("retrieve")
    r.add_argument("--index-dir", required=True)
    r.add_argument("--questions", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    r.add_argument("--no-rerank", action="store_true")
    r.add_argument("--no-editions", action="store_true", help="sans la gestion des éditions successives")

    g = sub.add_parser("generate")
    g.add_argument("--retrieved", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--backend", default="hf", choices=["hf", "openai", "echo"])
    g.add_argument("--tau", type=float, default=answer.TAU)
    g.add_argument("--no-check", action="store_true", help="sans ancrage numérique")
    g.add_argument("--run-name", default="pipeline")
    g.add_argument("--context", default="mixte", choices=["mixte", "chunks"])
    g.add_argument("--strategy", default="consensus", choices=["consensus", "preuve", "direct"])
    g.add_argument("--second-context", default="mixte", choices=["mixte", "chunks"])
    g.add_argument("--verify", action="store_true", help="second appel : le générateur relit sa preuve")

    args = ap.parse_args()
    if args.cmd == "index":
        index.build(args.pdf_dir, args.index_dir, args.chunking, not args.no_context, args.ocr)
    elif args.cmd == "update":
        index.update(args.index_dir, args.pdf_dir)
    elif args.cmd == "retrieve":
        answer.retrieve(args.index_dir, args.questions, args.out, args.mode, not args.no_rerank, not args.no_editions)
    else:
        answer.generate(args.retrieved, args.out, args.backend, args.tau, not args.no_check, args.run_name, args.context,
                        args.strategy, args.verify, args.second_context)


if __name__ == "__main__":
    main()
