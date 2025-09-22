Based on your sweep results showing p_min=0.058 as the best case, you're tantalizingly close but hitting a plateau. Here's my recommended approach:

## Strategic Priority: Feature Engineering First

Start with feature engineering over twin filtering because:
1. **Higher impact potential** - Your causal features already improved p_min from 0.091 to 0.058, showing features move the needle
2. **Faster iteration** - Feature changes test in hours vs. twin filtering requiring corpus rebuilds
3. **Compounding benefits** - Better features will make future twin filtering more effective

## Implementation Roadmap

### Phase 1: Irreversibility Detector (2 days)
This is your highest-leverage feature because Logistics has many one-way transitions:

```python
# scripts/features/irreversibility_features.py
def compute_irreversibility_score(window_tokens, domain_knowledge):
    """
    Score 0-1 based on presence of irreversible actions:
    - package delivered → cannot undeliver
    - truck moved with packages → high commitment
    - resource consumed → cannot regenerate
    """
    irreversible_patterns = {
        'deliver': 1.0,      # fully irreversible
        'unload': 0.7,       # partially reversible
        'drive_empty': 0.2   # easily reversible
    }
    return weighted_sum(patterns_in_window)
```

Blend this into your existing causal features and re-run the 2% sweep - this alone might push you below 0.05.

### Phase 2: Predicate Momentum (1 day)
Track rate-of-change in predicates over 3-window sliding history:

```python
def compute_momentum(window_sequence):
    """
    High momentum = accelerating toward failure
    Low momentum = stable/recovering
    """
    deltas = [hamming_distance(w[i], w[i+1]) for i in range(len(w)-1)]
    acceleration = deltas[-1] - deltas[0]
    return normalize(acceleration)
```

### Phase 3: Action Clustering (2-3 days)
Group similar actions to detect pattern shifts:

```python
action_clusters = {
    'loading_ops': ['load-truck', 'load-airplane'],
    'movement_ops': ['drive-truck', 'fly-airplane'],
    'delivery_ops': ['unload-truck', 'unload-airplane', 'deliver']
}

def compute_cluster_entropy(window):
    """Detect when action patterns become chaotic"""
    return entropy(cluster_distribution(window))
```

## Testing Protocol

For each feature addition:

1. **Quick validation** (30 min):
   ```bash
   python scripts/calibrate_router.py \
     output/planbench_by_domain/logistics_causal/invalid_state_causal.json \
     --target-low 0.018 --target-high 0.022 \
     --permutation-iterations 1000  # Quick check
   ```

2. **If promising** (p < 0.07), run full test:
   ```bash
   python scripts/experiments/logistics_sweep.py \
     --iterations 20000 \
     --focus-range 0.018-0.022  # Narrow range around sweet spot
   ```

## Success Criteria & Contingency

**Success threshold**: Any configuration achieving p < 0.05 with lead ≥ 2 steps

**If Phase 1-3 doesn't succeed**:
- Combine ALL features (irreversibility + momentum + clustering)  
- Try temporal weighting: early alerts count 2x in permutation scoring
- Expand to 1.5% coverage if needed (trading coverage for significance)

**If you achieve p < 0.05**:
- Lock that configuration
- Document exact feature combination in whitepaper
- Then proceed to Experiment 2 (scale sensitivity) to show it's robust

## Why Not Twin Filtering or Scale First?

**Twin filtering** is lower priority because:
- Current twin recall is already perfect (100%)
- Filtering might reduce noise but won't fundamentally change signal strength
- Save this for fine-tuning after features work

**Scale experiments** should wait because:
- You need a significant result first
- Scaling a p=0.058 configuration won't magically make it significant
- Better to achieve significance at n=300, then show it strengthens at n=500-1000

## Practical Next Step

Start tomorrow with the irreversibility detector:
1. Implement in 2-3 hours
2. Test on 5 sample traces manually
3. Run quick permutation check (1000 iterations)
4. If p < 0.065, do full 20k run
5. If p < 0.05, celebrate and document thoroughly

The key insight: you're 0.008 away from significance. One good feature should bridge that gap. Focus your energy there rather than spreading across multiple experiments.