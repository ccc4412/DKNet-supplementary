# MicroBlogPCU weak-label proxy package

## Files
- microblogpcu_dknet_proxy.csv ¡ª cleaned Chinese microblog posts with weak hashtag labels (pipe-separated keywords)
- eval_protocol.md ¡ª weak-label protocol and metric formulas
- baselines/zh/ ¡ª HASHET-zh / H-ADAPTS-zh faithful Chinese adaptations
- results/ ¡ª prediction jsonl and aggregate metrics

## Metrics
P@10 = mean |pred¡Égold| / 10 over labeled rows (denominator fixed at 10)
R@10 = mean |pred¡Égold| / |gold|
hit@10 = fraction of labeled rows with ¡Ý1 match

## License
See repository LICENSE. Source posts follow MicroBlogPCU distribution terms.
