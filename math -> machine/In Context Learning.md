---
aliases:
  - ICL
  - few shot learning
excalidraw-plugin: parsed
---
This is a new concept from the LLMs. It's where the model essentially **learns** without it's traditional sense of learning.
Well, a typical ML model **learns** by backpropagating and updating it's weights and parameters. And this happens during training (and also during fine tuning) - but not during inference.
But ICL is different, it is learning without updating the weights and biases.

"few-shot", "one-shot", "zero-shot" are just different variants of it - indicating how many examples we provide.

The idea is, it recognises patterns and then captures them and tries to find the same patterns in the new query.
