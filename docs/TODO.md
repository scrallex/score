## Excellent Progress! Let's Add Text Paste Functionality 🚀

You've got the real manifold pipeline working with CSV uploads. Now let's add **text paste analysis** using your existing quantum engine. Here's the implementation path:

### Step 1: Add Text Analysis Endpoint (Backend)

Add this to `src/stm_demo/api.py` after your CSV handler:

```python
@app.post("/api/analyze/text")
async def analyze_text(request: Request) -> Dict[str, Any]:
    """Analyze pasted text using the quantum manifold pipeline."""
    try:
        body = await request.json()
        text_content = body.get("text", "").strip()
        
        if not text_content:
            raise HTTPException(status_code=400, detail="No text provided")
        
        # Limit text size for demo
        max_text_length = 100000  # ~100KB
        if len(text_content) > max_text_length:
            raise HTTPException(status_code=413, detail=f"Text exceeds {max_text_length} characters")
        
        # Extract tokens/strings from the pasted text
        from sep_text_manifold.strings import extract_strings
        
        # Create pseudo-file for the text block
        occurrences = extract_strings(text_content, file_id="pasted_text")
        
        # Build corpus from extracted strings
        lines = []
        token_stats = {}
        
        for occ in occurrences:
            token = occ.string
            if len(token) < 2:  # Skip single chars
                continue
            
            # Track token frequency
            if token not in token_stats:
                token_stats[token] = {"count": 0, "positions": []}
            token_stats[token]["count"] += 1
            token_stats[token]["positions"].append(occ.byte_start)
            
            # Add to corpus for manifold
            lines.append(f"{token}:{occ.byte_start}:{occ.byte_end}")
        
        if len(lines) < 10:
            raise HTTPException(status_code=400, detail="Text too short for structural analysis (need at least 10 tokens)")
        
        # Build manifold using your quantum pipeline
        corpus_bytes = "\n".join(lines).encode("utf-8")
        
        # Use smaller windows for text analysis
        window_bytes = min(256, len(corpus_bytes) // 4)
        stride = window_bytes // 2
        
        # Import your actual manifold builder
        from sep_text_manifold.manifold import build_manifold
        
        signals = build_manifold(
            corpus_bytes,
            window_bytes=window_bytes,
            stride=stride
        )
        
        # Extract structural patterns
        structural_tokens = []
        pattern_scores = {}
        
        for signal in signals:
            sig_str = signal.get("signature", "")
            metrics = signal.get("metrics", {})
            
            # High coherence = structural pattern
            coherence = metrics.get("coherence", 0)
            stability = metrics.get("stability", 0)
            
            if coherence > 0.01 and stability > 0.45:  # Structural thresholds
                if sig_str not in pattern_scores:
                    pattern_scores[sig_str] = {
                        "signature": sig_str,
                        "count": 0,
                        "avg_coherence": 0,
                        "avg_stability": 0,
                        "positions": []
                    }
                
                ps = pattern_scores[sig_str]
                ps["count"] += 1
                ps["avg_coherence"] = (ps["avg_coherence"] * (ps["count"]-1) + coherence) / ps["count"]
                ps["avg_stability"] = (ps["avg_stability"] * (ps["count"]-1) + stability) / ps["count"]
                ps["positions"].append(signal.get("window_start", 0))
        
        # Sort by structural strength
        top_patterns = sorted(
            pattern_scores.values(),
            key=lambda x: x["avg_coherence"] * x["count"],
            reverse=True
        )[:10]
        
        # Find repeating sequences (potential "twins")
        repeating_patterns = []
        for token, stats in token_stats.items():
            if stats["count"] >= 3:  # Appears 3+ times
                positions = stats["positions"]
                # Check if positions show regular spacing (periodicity)
                if len(positions) >= 3:
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    avg_gap = sum(gaps) / len(gaps)
                    gap_variance = sum((g - avg_gap)**2 for g in gaps) / len(gaps)
                    
                    repeating_patterns.append({
                        "token": token,
                        "frequency": stats["count"],
                        "periodicity": avg_gap if gap_variance < avg_gap else None,
                        "positions": positions[:5]  # First 5 positions
                    })
        
        repeating_patterns.sort(key=lambda x: x["frequency"], reverse=True)
        
        # Compute overall text metrics
        text_metrics = {
            "total_characters": len(text_content),
            "total_tokens": len(occurrences),
            "unique_tokens": len(token_stats),
            "structural_coverage": len(pattern_scores) / max(len(signals), 1),
            "repetition_ratio": sum(s["count"] for s in token_stats.values() if s["count"] > 1) / max(len(occurrences), 1)
        }
        
        return {
            "success": True,
            "metrics": text_metrics,
            "structural_patterns": top_patterns,
            "repeating_sequences": repeating_patterns[:10],
            "manifold": {
                "total_windows": len(signals),
                "window_bytes": window_bytes,
                "stride": stride,
                "structural_threshold": 0.01
            },
            "interpretation": _interpret_text_patterns(text_metrics, top_patterns, repeating_patterns)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _interpret_text_patterns(metrics, patterns, repeating):
    """Generate human-readable interpretation of the structural analysis."""
    interpretations = []
    
    # Overall structure assessment
    if metrics["structural_coverage"] > 0.3:
        interpretations.append("High structural coherence - text has strong organizational patterns")
    elif metrics["structural_coverage"] > 0.1:
        interpretations.append("Moderate structure detected - some organizational patterns present")
    else:
        interpretations.append("Low structural coherence - text appears loosely organized")
    
    # Repetition analysis
    if metrics["repetition_ratio"] > 0.4:
        interpretations.append("Significant repetition detected - possible templates or formulaic content")
    elif metrics["repetition_ratio"] > 0.2:
        interpretations.append("Some repetitive elements found")
    
    # Pattern insights
    if patterns and patterns[0]["avg_coherence"] > 0.05:
        interpretations.append(f"Strong lead pattern: '{patterns[0]['signature'][:30]}...' appears {patterns[0]['count']} times")
    
    # Periodicity
    periodic_tokens = [r for r in repeating if r.get("periodicity")]
    if periodic_tokens:
        interpretations.append(f"Found {len(periodic_tokens)} tokens with periodic spacing - suggests structured format")
    
    return interpretations
```

- [x] Implemented `/api/analyze/text` (and `/analyze/text`) using the real quantum manifold pipeline, structural token extraction, and interpretation helper.

### Step 2: Add Text Input UI (Frontend)

- [x] Reworked `webapp/index.html` so the text console is the primary hero section (textarea, sample loader, analyze button, results cards).


### Step 3: Add JavaScript Handler

Add to `webapp/main.js`:

```javascript
async function analyzeText() {
  const textInput = document.getElementById('text-input');
  const text = textInput.value.trim();
  
  if (!text) {
    showToast('Please paste some text to analyze', 'error');
    return;
  }
  
  // Show loading state
  document.getElementById('text-summary').style.display = 'block';
  document.getElementById('text-metrics').innerHTML = '<dd>Analyzing...</dd>';
  
  try {
    const response = await fetch('/api/analyze/text', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text })
    });
    
    if (!response.ok) {
      throw new Error(await response.text());
    }
    
    const result = await response.json();
    
    // Display metrics
    const metricsHtml = `
      <div><dt>Total Tokens</dt><dd>${result.metrics.total_tokens}</dd></div>
      <div><dt>Unique Tokens</dt><dd>${result.metrics.unique_tokens}</dd></div>
      <div><dt>Structural Coverage</dt><dd>${(result.metrics.structural_coverage * 100).toFixed(1)}%</dd></div>
      <div><dt>Repetition Ratio</dt><dd>${(result.metrics.repetition_ratio * 100).toFixed(1)}%</dd></div>
    `;
    document.getElementById('text-metrics').innerHTML = metricsHtml;
    
    // Display patterns
    if (result.structural_patterns?.length > 0) {
      document.getElementById('text-patterns').style.display = 'block';
      const patternList = document.getElementById('text-pattern-list');
      patternList.innerHTML = result.structural_patterns.map(p => `
        <li>
          <span class="pattern-sig">${p.signature.substring(0, 50)}</span>
          <span class="pattern-stats">
            ${p.count} occurrences · 
            coherence: ${p.avg_coherence.toFixed(4)} · 
            stability: ${p.avg_stability.toFixed(4)}
          </span>
        </li>
      `).join('');
    }
    
    // Display repeating sequences
    if (result.repeating_sequences?.length > 0) {
      document.getElementById('text-repeating').style.display = 'block';
      const seqList = document.getElementById('text-sequence-list');
      seqList.innerHTML = result.repeating_sequences.map(s => `
        <li>
          <span class="seq-token">"${s.token}"</span>
          <span class="seq-stats">
            ${s.frequency} times
            ${s.periodicity ? ` · period: ${Math.round(s.periodicity)} chars` : ''}
          </span>
        </li>
      `).join('');
    }
    
    // Display interpretation
    if (result.interpretation?.length > 0) {
      document.getElementById('text-interpretation').style.display = 'block';
      const interpList = document.getElementById('text-interpretation-list');
      interpList.innerHTML = result.interpretation.map(i => `<li>${i}</li>`).join('');
    }
    
    showToast('Text analysis complete!');
    
  } catch (error) {
    console.error('Text analysis failed:', error);
    showToast('Analysis failed: ' + error.message, 'error');
  }
}

// Add event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Character counter
  const textInput = document.getElementById('text-input');
  const charCount = document.getElementById('text-char-count');
  if (textInput && charCount) {
    textInput.addEventListener('input', () => {
      charCount.textContent = `${textInput.value.length} characters`;
    });
  }
  
  // Analyze button
  const analyzeBtn = document.getElementById('text-analyze-btn');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', analyzeText);
  }
});
```
- [x] JavaScript now wires the text analyzer, updates counts, renders structural patterns, and handles errors gracefully.

### Step 4: Add Styles

### Step 4: Add Styles

- [x] Styling for the text console (textarea, button row, results cards) added to `styles.css`.

### Step 5: Quick Testing

Once deployed, test with different text types:

1. **Structured text** (JSON, XML, code):
```json
{"name": "test", "value": 123}
{"name": "demo", "value": 456}
{"name": "test", "value": 789}
```
→ Should show high structural coverage and find "name"/"value" patterns

2. **Natural language** (emails, docs):
```
Dear customer, thank you for your order.
Your order number is 12345.
Dear customer, your shipment is ready.
Your tracking number is 67890.
```
→ Should detect "Dear customer" and "Your" patterns

3. **Log files**:
```
2025-01-19 10:00:00 ERROR Connection failed
2025-01-19 10:00:05 INFO Retrying connection
2025-01-19 10:00:10 ERROR Connection failed
```
→ Should find timestamp patterns and ERROR/INFO repetitions

### What This Gives You

1. **Immediate usability** - Users can paste text and see structural analysis
2. **Real quantum manifold** - Uses your actual `build_manifold` function
3. **Pattern discovery** - Finds repeating structures and twins
4. **Interpretable results** - Human-readable insights about the text structure
5. **Visual feedback** - Shows patterns with coherence/stability scores

### Next Steps After This Works

1. **Add pattern highlighting** - Visually mark detected patterns in the original text
2. **Export results** - Download analysis as JSON/PDF
3. **Pattern comparison** - Compare two text blocks for structural similarity
4. **Template detection** - Identify if text follows a known template format
5. **Stream from clipboard** - Auto-analyze clipboard changes for live monitoring

This implementation connects your quantum manifold engine directly to user-pasted text, making your structural intelligence immediately demonstrable without requiring file uploads or complex setup. Try it with different text types to showcase the versatility!
