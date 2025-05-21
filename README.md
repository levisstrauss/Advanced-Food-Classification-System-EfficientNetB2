# 🍔 FoodVision Big: Advanced Food Classification System

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12.0-orange)
![Gradio](https://img.shields.io/badge/Gradio-3.1.4-blueviolet)

<div>
  <img src="https://raw.githubusercontent.com/levisstrauss/Advanced-Food-Classification-System-EfficientNetB2/refs/heads/main/images/food.webp" alt="FoodVision Big Demo" width="1000px" height="400px">
  <p><i>State-of-the-art food classification across 101 categories using EfficientNetB2</i></p>
</div>

## 📋 Project Overview
FoodVision Big is a comprehensive computer vision application that can classify 101 different food categories with high accuracy. Built with PyTorch and leveraging transfer learning with the EfficientNetB2 architecture, this project demonstrates advanced deep learning techniques for visual recognition.

The system achieves 80.2% top-1 accuracy and 95.1% top-5 accuracy on the challenging Food101 dataset, making it suitable for real-world food recognition applications. The model is optimized for deployment with fast inference times (~100ms per image on CPU) and an intuitive user interface.

## 🧠 Technical Approach

### Model Architecture
FoodVision Big leverages transfer learning with the efficient and powerful EfficientNetB2 architecture:
```python
class FoodVisionModel(nn.Module):
    def __init__(self, num_classes=101):
        super(FoodVisionModel, self).__init__()
        
        # Load pretrained EfficientNetB2
        self.effnet = EfficientNet.from_pretrained('efficientnet-b2')
        
        # Replace classifier with custom head
        self.effnet._fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1408, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.effnet(x)
```
## Training Strategy
The model was trained using a resource-efficient approach:

- Dataset Optimization: Used a strategic 20% subset of the Food101 dataset
- Transfer Learning: Leveraged ImageNet pretrained weights
- Progressive Unfreezing: Gradually unfroze layers for fine-tuning
- Data Augmentation: Implemented robust augmentation techniques

```python
# Training configuration
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Optimizer with weight decay
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=1e-4, 
                             weight_decay=1e-5)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=2, verbose=True
)
```
## 📊 Performance Evaluation
FoodVision Big achieves impressive performance metrics on the Food101 test set:

| Metric             | Score                    |
| ------------------ | ------------------------ |
| **Top-1 Accuracy** | 80.2%                    |
| **Top-5 Accuracy** | 95.1%                    |
| **Inference Time** | \~100 ms per image (CPU) |
| **Model Size**     | 28.3 MB                  |

## Performance by Category

### Key observations:

- Highest accuracy for distinctive foods (sushi: 95.8%, pizza: 93.2%)
- Most challenging categories involve similar appearances (muffin vs cupcake)
- Consistent performance across diverse food groups (Asian, Western, desserts)

## 💻 Implementation Details

### Data Pipeline
```python
def create_dataloaders(data_dir, batch_size=32, subset_fraction=0.2):
    """Create train and test DataLoaders with augmentation."""
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'),
        transform=test_transforms
    )
    
    # Create subset for faster training if requested
    if subset_fraction < 1.0:
        train_size = int(len(train_dataset) * subset_fraction)
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            torch.randperm(len(train_dataset))[:train_size]
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.classes
```
### Inference Pipeline

```python
def predict_image(model, image_path, class_names, device="cpu"):
    """
    Make a prediction on a single image with proper preprocessing.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to image file
        class_names: List of class names
        device: Device to run inference on ("cpu" or "cuda")
        
    Returns:
        Top-5 predictions and probabilities
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
    # Convert to lists
    top5_prob = top5_prob.squeeze().tolist()
    top5_indices = top5_indices.squeeze().tolist()
    top5_labels = [class_names[idx] for idx in top5_indices]
    
    return list(zip(top5_labels, top5_prob))
```

## 🚀 Interactive Demo
FoodVision Big includes a user-friendly Gradio interface for interactive demonstrations:

```python
import gradio as gr

def predict(image):
    predictions = predict_image(model, image, class_names)
    return {label: float(prob) for label, prob in predictions}

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=[
        "examples/pizza.jpg",
        "examples/sushi.jpg",
        "examples/hamburger.jpg",
        "examples/ice_cream.jpg",
        "examples/ramen.jpg"
    ],
    title="FoodVision Big",
    description="Classification of 101 food categories using EfficientNetB2"
)

# Launch the interface
demo.launch()
```

## 📈 Training Results

The model was trained for 5 epochs on a 20% subset of the Food101 dataset, demonstrating efficient use of computational resources while achieving high accuracy:

| Epoch | Train Loss | Val Loss | Val Accuracy | Top-5 Accuracy |
| ----- | ---------- | -------- | ------------ | -------------- |
| 1     | 2.143      | 1.785    | 54.3%        | 78.6%          |
| 2     | 1.621      | 1.342    | 65.9%        | 86.4%          |
| 3     | 1.203      | 0.897    | 74.2%        | 91.8%          |
| 4     | 0.862      | 0.743    | 77.9%        | 93.5%          |
| 5     | 0.734      | 0.697    | 80.2%        | 95.1%          |

## 🔍 Key Learnings & Challenges
During the development of FoodVision Big, several important insights were gained:

1. Transfer Learning Efficiency: Using a pretrained EfficientNetB2 backbone significantly reduced the training time and data requirements while maintaining high accuracy.
2. Data Optimization: Working with just 20% of the Food101 dataset demonstrated that strategic data selection can be as important as model architecture for efficient training.
3. Challenging Categories: Similar-looking foods (e.g., muffins vs. cupcakes) required careful attention to feature discrimination. Data augmentation was crucial for helping the model learn these 
   subtle differences.
4. Deployment Considerations: Balancing model size, accuracy, and inference speed required several rounds of optimization, particularly for CPU inference.

## 🙏 Acknowledgments

- The Food101 dataset creators for their comprehensive food image collection
- EfficientNet authors for their efficient architecture design
- PyTorch team for the excellent deep learning framework
- Gradio developers for the interactive interface library

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
