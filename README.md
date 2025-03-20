# 🍔 FoodVision Big

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.1.4-blueviolet)

**FoodVision Big** is an advanced computer vision application capable of classifying 101 different food categories with state-of-the-art accuracy using a fine-tuned EfficientNetB2 architecture.

<div align="center">
  <img src="./images/food.webp" alt="FoodVision Demo" width="600px" height="300px"/>
</div>

## 🚀 Features

- **High-Performance Model**: Fine-tuned EfficientNetB2 feature extractor delivering superior classification results
- **101 Food Categories**: Comprehensive food classification system
- **Fast Inference**: Optimized for quick prediction times
- **Interactive UI**: User-friendly Gradio interface for easy image uploads and real-time predictions
- **Deployment-Ready**: Structured for seamless deployment to Hugging Face Spaces

## 🔍 Model Architecture

FoodVision Big leverages transfer learning with an EfficientNetB2 backbone:

- **Base Model**: EfficientNetB2 pre-trained on ImageNet
- **Transfer Learning**: Feature extractor with custom classification head
- **Training Data**: Food101 dataset (20% subset for efficient training)
- **Performance**: Achieved remarkable accuracy with only 5 training epochs

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Top-1 Accuracy | 80.2% |
| Top-5 Accuracy | 95.1% |
| Inference Time | ~100ms per image on CPU |

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework
- **TorchVision**: Computer vision utilities and transforms
- **Gradio**: Interactive UI for model demonstrations
- **Python**: Core programming language

## 🧠 Project Insights

Creating FoodVision Big presented several challenges:

1. **Efficient Training**: Achieved optimal performance with limited computational resources by:
   - Using a 20% subset of the Food101 dataset
   - Implementing an effective feature extraction strategy
   - Employing strategic data augmentation

2. **Deployment Optimization**: Balanced model size and accuracy for real-world deployment scenarios

3. **UI/UX Design**: Created an intuitive interface with Gradio that showcases the model's capabilities

## 📈 Future Improvements

- Implement model quantization for faster inference
- Expand to mobile deployment with TorchScript/ONNX
- Integrate nutritional information database

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>Built with ❤️ by <a href="https://github.com/yourusername">Codemon</a></p>
</div>

