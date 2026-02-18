# Dimensional Aspect Sentiment Regression (DimASR)

## Overview

This project implements a fine-tuned BERT-based model for Dimensional Aspect Sentiment Regression.  
Given a sentence and a target aspect, the model predicts continuous Valence and Arousal scores in the range [1, 9].

## Task Definition

Input:
- Text
- One or more aspects

Output:
- Valence (1–9)
- Arousal (1–9)

## Approach

- Encoded input as: `Text [SEP] Aspect`
- Fine-tuned `bert-base-uncased`
- Used CLS embedding for regression
- Optimized with MSE loss and AdamW (lr=2e-5)

## Evaluation

Metrics:
- Development MSE Loss
- Pearson Correlation

Best Results:
- Valence Pearson: 0.88
- Arousal Pearson: 0.72

## Observations

- Strong performance for valence prediction
- Arousal modeling remains more challenging


