# CREMA-D Error Analysis Report

## 1. Overview
Total Samples: 2232

- **Errors**: 1800 (80.6%)
- **Correct**: 432 (19.4%)

## 2. Error Distribution by Type
- **General Error**: 1506 (83.7%)
- **Boundary Confusion**: 294 (16.3%)

## 3. Most Common Confusion Pairs
1. **sadness->fear**: 281 errors
2. **disgust->fear**: 277 errors
3. **anger->fear**: 253 errors
4. **neutral->fear**: 242 errors
5. **joy->fear**: 227 errors

## 4. Modality Bias Analysis
- **Balanced**: 1800 errors (100.0%)

## 5. Top-20 Surprising Misclassifications (Ranked by Confidence)

### Error #1
**Sample ID**: `call_2024-12-10_agent_01_001`
- **Ground Truth**: Anger
- **Predicted**: Neutral (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #2
**Sample ID**: `call_2024-12-19_agent_01_1370`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #3
**Sample ID**: `call_2024-12-14_agent_01_1315`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #4
**Sample ID**: `call_2024-12-13_agent_01_1314`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #5
**Sample ID**: `call_2024-12-10_agent_01_1311`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #6
**Sample ID**: `call_2024-12-18_agent_01_1309`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #7
**Sample ID**: `call_2024-12-17_agent_01_1308`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #8
**Sample ID**: `call_2024-12-13_agent_01_1304`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #9
**Sample ID**: `call_2024-12-12_agent_01_1303`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #10
**Sample ID**: `call_2024-12-18_agent_01_1299`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #11
**Sample ID**: `call_2024-12-17_agent_01_1298`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #12
**Sample ID**: `call_2024-12-16_agent_01_1297`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #13
**Sample ID**: `call_2024-12-15_agent_01_1296`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #14
**Sample ID**: `call_2024-12-11_agent_01_1292`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #15
**Sample ID**: `call_2024-12-10_agent_01_1291`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #16
**Sample ID**: `call_2024-12-16_agent_01_1287`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #17
**Sample ID**: `call_2024-12-15_agent_01_1286`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #18
**Sample ID**: `call_2024-12-14_agent_01_1285`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #19
**Sample ID**: `call_2024-12-13_agent_01_1284`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

### Error #20
**Sample ID**: `call_2024-12-12_agent_01_1283`
- **Ground Truth**: Neutral
- **Predicted**: Fear (Confidence: 0.80)
- **Attention Weights**: Audio 0.50, Text 0.50
- **Error Type**: General Error
- **Analysis**: Model over-relied on balanced cues.

## 6. Insights & Recommendations

### 6.1 Confusion Patterns
- **sadness->fear**: 281 cases. These emotions share low arousal characteristics. Consider adding energy-based features.
- **disgust->fear**: 277 cases. Review acoustic and linguistic patterns for this pair.
- **anger->fear**: 253 cases. Review acoustic and linguistic patterns for this pair.


### 6.2 Modality Recommendations
- Modality balance is good. No immediate attention mechanism adjustments needed.


### 6.3 Action Items
1. Implement class weighting for underperforming emotions.
2. Balance audio-text attention mechanism (e.g. Entropy-based gating).
3. Add arousal-based features to distinguish similar emotions.
4. Validate ground truth labels for high-confidence errors.
