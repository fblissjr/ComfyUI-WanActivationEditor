# Embedding Amplifier Guide

## The Problem It Solves

When using FP8 quantization, even "different" prompts often produce embeddings that are only 10-30% different. This is too similar for visible injection effects. The Embedding Amplifier ensures your injection embeddings are different enough to work.

## How to Use It

1. **Connect Your Embeddings**
   - Main prompt → `main_embeds` input
   - Injection prompt → `injection_embeds` input

2. **Set Target Difference**
   - Default: 60% (good for most cases)
   - Range: 30-90%
   - Higher = more dramatic effects

3. **Choose Amplification Mode**
   - `push_apart`: Gentle amplification, preserves structure (default)
   - `maximize_diff`: Aggressive, focuses on high-variance dimensions
   - `orthogonalize`: Mathematical approach, makes embeddings perpendicular

4. **Use Amplified Output**
   - Connect `amplified_injection` to WanVideoActivationEditor's injection input

## Example Workflow

```
Main Prompt: "A sunny beach with people"
Injection Prompt: "A rainy beach with people"
Original Difference: 22%  ← Too similar!

→ Amplifier (60% target) →

Amplified Difference: 60%  ← Now it works!
```

## Tips

- Start with `push_apart` mode - it's the safest
- If effects are still subtle, try 70-80% target
- Use `maximize_diff` for more dramatic changes
- `orthogonalize` is best for style transfer

## When NOT to Use It

- If your prompts already differ by >50%, you don't need it
- If you want subtle effects, keep target low (30-40%)
- Don't use with completely unrelated prompts (they're already different enough)