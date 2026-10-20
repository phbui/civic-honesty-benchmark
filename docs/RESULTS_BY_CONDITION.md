# Results by condition, every setting

Neutral, level-described, and increment-described conditions for all ten settings, base regime. Abstain and fabricate are shares of reported unanswerable episodes; accuracy is over answered answerable items; caveat is the share of answered unreliable items whose uncertainty statement states the documented R of about 0.50; disclosure is the strict tier over eligible answered items. The level and increment conditions are exploratory and run in settings other than Claude Haiku 4.5 on the same questions. Recompute with `scripts/summarize_eval.py` and `scripts/disclosure.py`.

| Setting | Condition | Abstain | Fabricate | Accuracy | Caveat | Disclosure |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | neutral | 96.2% | 3.8% | 89.2% | 100.0% | 0/108 |
| Claude Haiku 4.5 | level | 92.3% | 7.7% | 90.1% | 100.0% | 0/112 |
| Claude Haiku 4.5 | increment | 91.2% | 8.8% | 89.3% | 100.0% | 1/108 |
| Claude Haiku 5.5 | neutral | 99.5% | 0.5% | 89.4% | 100.0% | 8/112 |
| Claude Haiku 5.5 | level | 99.5% | 0.5% | 88.4% | 100.0% | 8/112 |
| Claude Haiku 5.5 | increment | 99.5% | 0.5% | 88.4% | 100.0% | 7/112 |
| Claude Sonnet 5.5 | neutral | 100.0% | 0.0% | 85.5% | 100.0% | 8/62 |
| Gemini 3.5 Flash-Lite | neutral | 96.3% | 3.7% | 92.0% | 100.0% | 1/113 |
| Gemini 3.5 Flash-Lite | level | 95.9% | 4.1% | 89.4% | 100.0% | 2/113 |
| Gemini 3.5 Flash-Lite | increment | 96.0% | 4.0% | 89.5% | 100.0% | 0/113 |
| gpt-5-mini, minimal | neutral | 73.8% | 26.2% | 76.3% | 100.0% | 33/103 |
| gpt-5-mini, minimal | level | 71.9% | 28.1% | 74.8% | 100.0% | 39/104 |
| gpt-5-mini, minimal | increment | 66.2% | 33.8% | 80.7% | 100.0% | 37/102 |
| gpt-5-mini, low | neutral | 100.0% | 0.0% | 71.0% | 100.0% | 35/105 |
| gpt-5-mini, low | level | 99.0% | 1.0% | 74.7% | 100.0% | 24/107 |
| gpt-5-mini, low | increment | 96.9% | 3.1% | 75.5% | 100.0% | 30/106 |
| gpt-oss-120b | neutral | 98.5% | 1.5% | 85.2% | 100.0% | 1/44 |
| gpt-oss-120b | level | 98.6% | 1.4% | 78.8% | 100.0% | 0/36 |
| gpt-oss-120b | increment | 100.0% | 0.0% | 84.6% | 100.0% | 0/48 |
| Qwen3-32B | neutral | 97.5% | 2.5% | 84.1% | 100.0% | 0/99 |
| Qwen3-32B | level | 98.5% | 1.5% | 82.2% | 100.0% | 0/94 |
| Qwen3-32B | increment | 97.5% | 2.5% | 80.5% | 100.0% | 0/90 |
| GLM-5.3-Flash | neutral | 100.0% | 0.0% | 95.7% | 100.0% | 18/88 |
| GLM-5.3-Flash | level | 100.0% | 0.0% | 93.3% | 100.0% | 30/84 |
| GLM-5.3-Flash | increment | 100.0% | 0.0% | 92.0% | 100.0% | 17/82 |
| Qwen3.8-Flash | neutral | 100.0% | 0.0% | 98.6% | 100.0% | 14/47 |
| Qwen3.8-Flash | level | 100.0% | 0.0% | 97.1% | 100.0% | 12/43 |
| Qwen3.8-Flash | increment | 100.0% | 0.0% | 93.8% | 100.0% | 15/39 |
