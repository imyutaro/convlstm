### memo for implementation with keras.

#### model definition
```python
input_shape = (None, 599, 128)
inputs = Input(batch_shape=input_shape)
print("batch_shape=input_shape : ", inputs) # -> Tensor("input_1:0", shape=(?, 599, 128), dtype=float32)
inputs = Input(input_shape)
print("just input_shape : ", inputs) # -> Tensor("input_2:0", shape=(?, ?, 599, 128), dtype=float32)
```
