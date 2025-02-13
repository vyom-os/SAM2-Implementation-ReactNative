### Segment Anything Model 2 in React Native
Segment Anything Model 2 (SAM 2) is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect our SA-V dataset, the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

__Download SAM2__ onnx model from https://huggingface.co/models?p=1&sort=trending&search=segment+anything
Preferabley generate it yourself. Follow the steps given in https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#onnx-export

__ONNX Runtime__ is a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries. ONNX Runtime can be used with models from PyTorch, Tensorflow/Keras, TFLite, scikit-learn, and other frameworks. Checkout https://onnxruntime.ai/docs/

Follow the microsoft's onnxruntime-inference-examples for better understanding of onnx runtime. https://github.com/microsoft/onnxruntime-inference-examples


__Understanding the working__
Segment Anything Model comes in two parts. One is encoder, that encodes and prepares the image for segmentation. Other is decoder that takes the points and encoder result to produce the mask. 

SAM is trained with images of dimension 1024X1024. So the model scales the image to 1024X1024 format before working on it.  Keeping this in mind you should mindfully give the input points of the image(scaled to 1024). 

The decoder produces multiple masks with different scores(ranging between 0 and 1). The score is the models confidence on that mask. Each point in a mask is a set of x and y coordinate of image.  Also each point has its own value(ranging between 0 and 1) again representing model’s confidence on that point. You can filter the mask points based on these scores.

For faster inference and to avoid inturruption to UI thread we have used kotlin's coroutines to load models and process image. We created a native module and exposed it to React Native component. 

__Note__: For selecting multiple points just keep on adding the points in the points array with label 1. For removing a segment send a point of that segment with label 0.


__Follow steps to run the app__

> git clone https://github.com/vyom-os/SAM2ImplementationReactNative.git

> Create an “assets” folder in android/app/src/main/assets. Paste the onnx models there. 

> npm i      //Installing the packages

> npx react-native run-android     //run the app



