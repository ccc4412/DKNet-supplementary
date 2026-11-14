# Weak-label protocol

1. Input text has hashtags stripped; hashtags appear only as evaluation labels.
2. Labeled rows: ~15% of posts; mean labels/row ≈ 1.31; vocab ≈ 2570.
3. All methods share the same candidate pools and top-K=10 truncation.
4. HASHET-zh: as-of retrain on weak-labeled history; H-ADAPTS-zh: window=300, K=10, τ=0.9.
5. DKNet proxy numbers are from proxy_hybrid_* ablations in proxy_real_data/.
