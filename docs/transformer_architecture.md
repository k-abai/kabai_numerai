# Transformer Architecture for Numerai

## Overview
This model projects scalar features into tokens, processes them via Transformer Encoder blocks, and outputs a prediction through an MLP head.

## Diagram
```mermaid
graph TD
    A[Input Features] --> B[Feature Embedding]
    B --> C[Positional Encoding]
    C --> D[Transformer Encoder Block 1]
    D --> E[Transformer Encoder Block 2]
    E --> F[Global Average Pooling]
    F --> G[MLP Head]
    G --> H[Prediction]
```
