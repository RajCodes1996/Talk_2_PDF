Part A: Preflight
python -c "import rank_bm25, sklearn; print('ok')"
ok
python -m pytest -q | Select-Object -Last 3
..................                                                       [100%]
18 passed in 86.08s (0:01:26)
git status --short
fatal: not a git repository (or any of the parent directories): .git
Part C: Verification (Items 1 to 5)
1. Unit Tests
python -m pytest tests/test_retrieval.py -q
........                                                                 [100%]
8 passed in 8.40s
2. Full Test Suite
python -m pytest -q | Select-Object -Last 3
..........................                                               [100%]
26 passed in 155.22s (0:02:35)
3. Chunk Sanity on data/samples/test_document_5pages.pdf
python -c "from src.ingestion.extractor import extract_document as e; from src.retrieval.chunker import build_chunks as b; c=b(e(r'data\samples\test_document_5pages.pdf')); print(len(c)); [print(x['page_start'], x['page_end'], x['n_words'], x['heading'][:30]) for x in c]"
5
1 1 38 Screen Readers
2 2 30 Speech Rate
3 3 34 Visual Contrast
4 4 27 Braille Displays
5 5 19 Keyboard Navigation
4. Three CLI Queries
python -m src.retrieval.cli data\samples\test_document_5pages.pdf "Is NVDA free?"
5 chunks
[p.1] Screen Readers (score 1.0): NVDA is a free and open source screen reader for Windows. It converts text on the screen into speech or braille output. Many blind users rely on it every day for web browsing and document reading acro
python -m src.retrieval.cli data\samples\test_document_5pages.pdf "What speaking speed do beginners prefer?"
5 chunks
[p.2] Speech Rate (score 1.0): Beginners usually prefer a slower speech rate of about 150 words per minute. Experienced users often listen at more than 300 words per minute without losing comprehension of the content.
python -m src.retrieval.cli data\samples\sam.pdf "What is this document about?" -k 2
2 chunks
[p.1] The 4Ds: Diligence (score 0.5): Diligence is taking responsibility for what we do with AI and how we do it. “In this document, We did good. Claude 3.7 ...”
5. Classifier Path Test
python -m src.retrieval.cli data\samples\test_document_5pages.pdf "Is NVDA free?" --classify
5 chunks
[p.1] Screen Readers (score 1.0): NVDA is a free and open source screen reader for Windows. It converts text on the screen into speech or braille output. Many blind users rely on it every day for web browsing and document reading acro
Part E: Evaluation Output
python -m tools.eval_retrieval --sweep
questions=22 docs=7 chunks={'test_document_5pages': 5, 'sam': 2, 'mixed_test_3pages': 8, 'nutrition': 4, 'energy': 4, 'history': 4, 'finance': 4}
alpha=0.0: hit@1=0.864 hit@3=1.000 MRR=0.932
alpha=0.3: hit@1=0.864 hit@3=1.000 MRR=0.932
alpha=0.5: hit@1=0.864 hit@3=1.000 MRR=0.932
alpha=0.7: hit@1=0.818 hit@3=1.000 MRR=0.909
alpha=1.0: hit@1=0.409 hit@3=0.455 MRR=0.432
python -m tools.eval_retrieval --classify
questions=22 docs=7 chunks={'test_document_5pages': 5, 'sam': 1, 'mixed_test_3pages': 9, 'nutrition': 8, 'energy': 8, 'history': 8, 'finance': 8}
alpha=0.5: hit@1=1.000 hit@3=1.000 MRR=1.000

Decision rule: Kept default alpha=0.5 (no other alpha improved MRR by >= 0.03).

Part F: Git Commit
git log --oneline -3
1063f2e Day 8: hybrid BM25+TF-IDF retrieval with page citations
Deviations
Created data/samples/test_document_5pages.pdf using PyMuPDF to satisfy the Part C item 3 test file requirement.
Initialized local git repository (git init) to commit changes as requested in Part F.