This demo(based on @greyovo's [jupyter notebook](https://colab.research.google.com/drive/1bW1aMg0er1T4aOcU5pCNYVgmVzBJ4-x4#scrollTo=hPscj2wlZlHb)) show the inference result for the same text(Chinese & English) and image input with different model, including the [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP/tree/master) and the onnx quantized model. Test result on my local machine is as follows:
| model |   | result |
|---|---|---|
| [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP/tree/master) | Chinese | [[1.9532440e-03 9.9525285e-01 2.2442457e-03 5.4962368e-04]] |
| [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP/tree/master) | English | [[2.5376787e-03 9.9683857e-01 4.3544930e-04 1.8830669e-04]] |
| clip-cn-image-encoder.onnx & clip-cn-text-encoder.onnx | Chinese | [[1.9535627e-03 9.9525201e-01 2.2446462e-03 5.4973643e-04]] |
| clip-cn-image-encoder.onnx & clip-cn-text-encoder.onnx | English | [[2.5380836e-03 9.9683797e-01 4.3553708e-04 1.8835040e-04]] |
| clip-cn-image-encoder-quant-int8.onnx & clip-cn-text-encoder-quant-int8.onnx | Chinese | [[0.00884504 0.98652565 0.00179121 0.00283814]] |
| clip-cn-image-encoder-quant-int8.onnx & clip-cn-text-encoder-quant-int8.onnx | English | [[0.02240802 0.97132427 0.00435637 0.00191139]] |

The test English input text is ["a tiger", "a cat", "a dog", "a bear"], Chinese input text is ["老虎", "猫", "狗", "熊"], and the test image is as follows:
![](image.jpg)