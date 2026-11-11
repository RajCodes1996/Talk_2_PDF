PS C:\Users\netware\Desktop\vision> python -m pytest tests/test_retrieval.py -q
>> python -m src.retrieval.cli data\samples\sam.pdf "What is transparency diligence?" -k 2
>> python -m tools.eval_retrieval --exclude test_document_5pages --out data\eval\results_nocls.txt | Select-Object -Last 10
.........                                                                                                                          [100%] 
=========================================================== warnings summary ============================================================ 
..\..\AppData\Roaming\Python\Python314\site-packages\anyio\pytest_plugin.py:14
  C:\Users\netware\AppData\Roaming\Python\Python314\site-packages\anyio\pytest_plugin.py:14: PytestRemovedIn10Warning: _pytest.python.CallSpec2 has been renamed to CallSpec.
  The CallSpec2 alias will be removed in pytest 10.
  Update imports to use CallSpec instead.
  See https://docs.pytest.org/en/stable/deprecations.html#callspec2-renamed
    from _pytest.python import CallSpec2

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
9 passed, 1 warning in 9.52s
2 chunks
[p.1] Creation Diligence; Transparency Diligence; Deployment Diligence (score 0.5): Being thoughtful about which AI systems you use and how you interact with them. Being honest about AI's role in your work with everyone who needs to know. Taking responsibility for verifying and vouch
[p.1] The 4Ds: Diligence (score 0.3341): Diligence is taking responsibility for what we do with AI and how we do it. We did good. “In this document, Claude 3.7 ...”
questions=17 docs=6 chunks={'sam': 2, 'mixed_test_3pages': 8, 'nutrition': 4, 'energy': 4, 'history': 4, 'finance': 4}
alpha=0.5: hit@1=0.941 hit@3=1.000 MRR=0.971
not at rank 1, alpha=0.5 (doc | question | rank | top1 page):
  sam | Who holds the copyright for the 4Ds document? | 2 | 1