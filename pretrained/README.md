## Pretrained Models Summary

| Name | Train Loss | Test Loss |#parameters|#epochs|#hidden_channels|lr|Comments|
|--|--|--|--|--|--|--|--|
|[a StateGNNEncoderConvEdgeAttrSPA](./models/StateGNNEncoderConvEdgeAttrSPA/)StateGNNEncoderConvEdgeAttrSPA  |0.006074  |0.006214 |12,392|20|32|0.0001||
|StateGNNEncoderConvEdgeAttrBasic32Ch  |0.007862  |0.007962 |9,896|20|32|0.0001||
|StateGNNEncoderConvEdgeAttrBasic  |0.007512  |0.007499 |36,168|20|64|0.0001||

## Inference
model = StateModelEncoder(hidden_channels=#hidden_channels, out_channels=8)

