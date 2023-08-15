## Pretrained Models Summary

| Name | Train Loss | Test Loss |#parameters|#epochs|#hidden|lr|Comments|
|--|--|--|--|--|--|--|--|
|[StateGNNEncoderConvEdgeAttrSPA](./models/StateGNNEncoderConvEdgeAttrSPA/)  |0.006074  |0.006214 |12,392|20|32|0.0001||
|[StateGNNEncoderConvEdgeAttrBasic32Ch](./models/StateGNNEncoderConvEdgeAttrBasic32Ch/)  |0.007862  |0.007962 |9,896|20|32|0.0001||
|[StateGNNEncoderConvEdgeAttrBasic](./models/StateGNNEncoderConvEdgeAttrBasic/)  |0.007512  |0.007499 |36,168|20|64|0.0001||
|[StatesAfterAllCompact32ch](./models/StatesAfterAllCompact32ch/)  |0.006934  |0.006977 |8,296|20|32|0.0001||
|[StatesAfterAllCompact32ch](./models/StatesAfterAllCompact32ch/)  |0.006135  |0.006215 |8,296|50|32|0.0001||

## Inference
```python
model = StateModelEncoder(hidden_channels=#hidden, out_channels=8)
```
