"""Command line interface for Sanad."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import ingest_corpus
from .embed import get_embedder
from .evaluate import (evaluate_answers, evaluate_retrieval, load_questions,
                       report_markdown, validate_dataset)
from .generate import ChatClient, OllamaChat
from .pipeline import RagEngine, Status
from .retrieve import HybridRetriever
from .store import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "sanad.db"
DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_EVAL = ROOT / "eval" / "questions.jsonl"


def _open_store(args) -> Store:
    db = Path(args.db)
    if getattr(args, "reset", False):
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    return Store(db)


def _retriever(store: Store, args) -> HybridRetriever:
    embedder = get_embedder(args.embedder, model=args.embed_model,
                            base_url=args.ollama_url)
    return HybridRetriever(store, embedder, candidates=args.candidates)


def _chat(args) -> ChatClient:
    return OllamaChat(model=args.model, base_url=args.ollama_url)


def cmd_ingest(args) -> int:
    with _open_store(args) as store:
        embedder = get_embedder(args.embedder, model=args.embed_model,
                                base_url=args.ollama_url)
        t0 = time.perf_counter()
        stats = ingest_corpus(store, embedder, args.corpus, force=args.force,
                              max_chars=args.max_chars)
        elapsed = time.perf_counter() - t0
        for s in stats:
            state = "skipped (unchanged)" if s.skipped else f"{s.chunks} chunks"
            print(f"{s.doc_id:<20} {state}")
        total = store.stats()
        print(f"\ningested {len(stats)} docs in {elapsed:.1f}s — "
              f"{total['documents']} docs, {total['chunks']} chunks, "
              f"{total['vectors']} vectors ({total['embedding_model']})")
    return 0


def cmd_search(args) -> int:
    with _open_store(args) as store:
        hits = _retriever(store, args).search(
            args.query, top_k=args.k, lang=args.lang, mode=args.mode)
    if args.json:
        print(json.dumps([
            {"id": h.chunk.id, "title": h.chunk.title, "section": h.chunk.section,
             "lang": h.chunk.lang, "bm25": h.bm25_score, "dense": h.dense_score,
             "rrf": round(h.score, 6), "text": h.text[:300]}
            for h in hits], ensure_ascii=False, indent=2))
    else:
        for i, h in enumerate(hits, start=1):
            signals = []
            if h.bm25_rank:
                signals.append(f"bm25 #{h.bm25_rank}")
            if h.dense_rank:
                signals.append(f"dense #{h.dense_rank}")
            print(f"{i}. [{h.chunk.id}] {h.chunk.title} — {h.chunk.section} "
                  f"({', '.join(signals)})")
            print(f"   {h.text[:240]}")
    return 0 if hits else 3


def cmd_ask(args) -> int:
    with _open_store(args) as store:
        engine = RagEngine(
            _retriever(store, args), _chat(args),
            support_threshold=args.support_threshold,
            dense_min_score=args.dense_min_score)
        result = engine.ask(args.question, top_k=args.k, lang=args.lang, mode=args.mode)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        labels = {Status.ANSWERED: "ANSWERED", Status.INSUFFICIENT: "ABSTAINED",
                  Status.UNGROUNDED: "UNGROUNDED"}
        print(f"[{labels[result.status]}] {result.answer or result.reason}")
        if result.grounding:
            print(f"grounding: {result.grounding.grounded_ratio:.2f} "
                  f"({result.grounding.grounded}/{len(result.grounding.scorable)} sentences)")
        if args.show_sources:
            print("\nSources:")
            for i, s in enumerate(result.sources, start=1):
                print(f"  [S{i}] {s.chunk.id} — {s.chunk.title} / {s.chunk.section}")
        print(f"\n{result.retrieval_ms + result.generation_ms:.0f} ms "
              f"(retrieval {result.retrieval_ms:.0f} ms), model {result.model}")
    return 0 if result.status is Status.ANSWERED else 3


def cmd_validate(args) -> int:
    questions = load_questions(args.eval_file)
    errors = validate_dataset(questions, args.corpus)
    if errors:
        for e in errors:
            print(f"INVALID {e}")
        return 1
    dev = sum(1 for q in questions if q.split == "dev")
    print(f"dataset OK: {len(questions)} questions (dev {dev}, test {len(questions) - dev}), "
          f"{sum(1 for q in questions if not q.answerable)} unanswerable")
    return 0


def cmd_eval(args) -> int:
    questions = load_questions(args.eval_file)
    errors = validate_dataset(questions, args.corpus)
    if errors:
        for e in errors:
            print(f"INVALID {e}")
        return 1

    with _open_store(args) as store:
        retriever = _retriever(store, args)
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        retrieval_reports = [
            evaluate_retrieval(retriever, questions, k=args.k, mode=mode, split=args.split)
            for mode in modes
        ]
        answer_report = None
        if args.generate:
            engine = RagEngine(
                retriever, _chat(args),
                support_threshold=args.support_threshold,
                dense_min_score=args.dense_min_score)
            answer_report = evaluate_answers(engine, questions, k=args.k, split=args.split)

        stats = store.stats()
        dataset = {
            "path": str(args.eval_file),
            "n_questions": len(questions),
            "n_dev": sum(1 for q in questions if q.split == "dev"),
            "n_test": sum(1 for q in questions if q.split == "test"),
            "corpus_documents": stats["documents"],
            "corpus_chunks": stats["chunks"],
            "embedding_model": stats["embedding_model"],
        }

    payload = {
        "dataset": dataset,
        "retrieval": [r.to_dict() for r in retrieval_reports],
        "answers": answer_report.to_dict() if answer_report else None,
    }
    md = report_markdown(retrieval=retrieval_reports, answers=answer_report, dataset=dataset)
    print(md)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(md + "\n", encoding="utf-8")
        print(f"wrote {args.md_out}")
    return 0


def cmd_stats(args) -> int:
    with _open_store(args) as store:
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sanad", description=__doc__)
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--embedder", choices=["ollama", "hashing"], default="ollama")
    p.add_argument("--embed-model", default="bge-m3")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--candidates", type=int, default=30)
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="index the corpus snapshot")
    ing.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ing.add_argument("--force", action="store_true", help="re-embed unchanged documents")
    ing.add_argument("--reset", action="store_true", help="delete the local database first")
    ing.add_argument("--max-chars", type=int, default=900)
    ing.set_defaults(func=cmd_ingest)

    srch = sub.add_parser("search", help="retrieve sources without an LLM")
    srch.add_argument("query")
    srch.add_argument("-k", type=int, default=5)
    srch.add_argument("--lang", choices=["ar", "en"])
    srch.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    srch.add_argument("--json", action="store_true")
    srch.set_defaults(func=cmd_search)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5)
    ask.add_argument("--lang", choices=["ar", "en"])
    ask.add_argument("--mode", choices=["hybrid", "lexical"], default="hybrid")
    ask.add_argument("--support-threshold", type=float, default=0.6)
    ask.add_argument("--dense-min-score", type=float, default=None)
    ask.add_argument("--json", action="store_true")
    ask.add_argument("--show-sources", action="store_true")
    ask.set_defaults(func=cmd_ask)

    val = sub.add_parser("validate", help="check gold labels against the corpus")
    val.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    val.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    val.set_defaults(func=cmd_validate)

    ev = sub.add_parser("eval", help="run retrieval and end-to-end evaluation")
    ev.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ev.add_argument("--eval-file", default=str(DEFAULT_EVAL))
    ev.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ev.add_argument("-k", type=int, default=5)
    ev.add_argument("--modes", default="hybrid,lexical")
    ev.add_argument("--generate", action="store_true", help="also call the LLM")
    ev.add_argument("--support-threshold", type=float, default=0.6)
    ev.add_argument("--dense-min-score", type=float, default=None)
    ev.add_argument("--json-out", default="")
    ev.add_argument("--md-out", default="")
    ev.set_defaults(func=cmd_eval)

    st = sub.add_parser("stats", help="show index statistics")
    st.set_defaults(func=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
