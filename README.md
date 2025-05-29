### Segment Anything Model 2 in React Native(Android part)
Segment Anything Model 2 (SAM 2) is a foundation model aimed at solving promptable visual segmentation in images and videos. By integrating Vyom with SAM2, we enable real-time image processing and machine learning image segmentation capabilities, delivering AI image segmentation solutions that power real-time data analysis. Meta extended SAM2 to video by considering images as a single-frame video, leveraging SAM2 image processing to unify image segmentation and other critical applications.

__Download SAM2__ onnx model from https://huggingface.co/models?p=1&sort=trending&search=segment+anything
Preferabley generate it yourself. Follow the steps given in https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#onnx-export

**ONNX Runtime** is a high-performance cross-platform machine learning model accelerator, enabling integration with various hardware libraries. **ONNX Runtime for deep learning models** supports models from frameworks like PyTorch, TensorFlow/Keras, TFLite, scikit-learn, and more. We use **ONNX Runtime integration for mobile apps** to accelerate the model inference, particularly in the context of **real-time AI inference for mobile devices**.

For more details on **ONNX Runtime**, visit the official [ONNX Runtime Documentation](https://onnxruntime.ai/docs/). Also, check out the [Microsoft ONNX Runtime Inference Examples](https://github.com/microsoft/onnxruntime-inference-examples) for further understanding of ONNX's functionality.


__Understanding the working__
The **Segment Anything Model** consists of two parts: the **encoder** and the **decoder**. The encoder prepares and encodes the image for segmentation, while the decoder processes the points from the encoder's output to produce the segmentation masks.

SAM2 is trained with images of dimension **1024x1024**. The model scales the image to this format before performing any processing. It’s crucial to provide input points scaled to **1024x1024** for accurate results.

The decoder produces multiple masks, each associated with a score (ranging between 0 and 1). The score reflects the model's confidence in each mask, with each point in the mask representing its coordinates in the image. Each point also carries a confidence score, allowing you to filter **low-confidence masks or points** when needed.

For faster inference and to avoid interruptions to the UI thread, we use **Kotlin Coroutines for machine learning**. This ensures asynchronous loading of the **SAM2 encoder and decoder ONNX files** without blocking the React Native UI thread. We created a native module in **React Native with ONNX Runtime** to expose this segmentation functionality to the React Native components for seamless integration.

**Note**: For selecting multiple points, keep adding the points in the points array with label 1. To remove a segment, add a point of that segment with label 0.


__Follow steps to run the app__

> git clone https://github.com/vyom-os/SAM2ImplementationReactNative.git

> Create an “assets” folder in android/app/src/main/. Paste the onnx models there. 

> npm i      //Installing the packages


By incorporating **Vyom integration** with **SAM2 model**, we create powerful **AI-powered mobile apps for segmentation**, enhancing real-time **image analysis for mobile** and boosting the capabilities of **object segmentation in mobile apps**. Using ONNX runtime not only improved the performance of mobile apps but also ensured smooth **real-time data analysis with SAM2**, enabling users to carry out **AI image segmentation** tasks with **real-time decision-making**.

Feel free to explore the integration of **SAM2 (Segment Anything Model 2)** in **React Native** and dive into our **open-source AI model integration** for advanced **image segmentation** and other critical use cases.
