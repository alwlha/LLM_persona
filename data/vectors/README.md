# Persona Vector Files

Put Big Five trait vectors computed from `assistant-axis` here.

Recommended layout for Qwen3-8B:

```text
data/vectors/qwen3-8b/
  openness.pt
  conscientiousness.pt
  extraversion.pt
  agreeableness.pt
  neuroticism.pt
```

Each file can be either:

- a raw tensor with shape `[num_layers, hidden_size]`, or
- a dict containing `{"vector": tensor}`.

You can also use a single bundled `.pt` file and set `meta.trait_vectors`
to that file path in `data/activation/vector.json`.
