# MicroBlogPCU weak-label proxy package

## Files
- microblogpcu_dknet_proxy.csv �� cleaned Chinese microblog posts with weak hashtag labels (pipe-separated keywords)
- eval_protocol.md �� weak-label protocol and metric formulas
- baselines/zh/ �� HASHET-zh / H-ADAPTS-zh faithful Chinese adaptations
## Metrics
P@10 = mean |pred��gold| / 10 over labeled rows (denominator fixed at 10)
R@10 = mean |pred��gold| / |gold|
hit@10 = fraction of labeled rows with ��1 match

## License
See repository LICENSE. Source posts follow MicroBlogPCU distribution terms.
