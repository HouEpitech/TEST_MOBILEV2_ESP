import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

class_names = {
    0: "classe_1",
    1: "Apple",
    2: "classe_3"
}


def detect_image(image_path, model_path):
    model = load_model(model_path)

    image = cv2.imread(image_path)
    if image is None:
        print("图片未找到")
        return None

    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)

    predictions = model.predict(image_expanded)

    # 显示所有类别的预测概率
    for class_idx, probability in enumerate(predictions[0]):
        class_name = class_names.get(class_idx, f"未知类别_{class_idx}")
        print(f"{class_name}: {probability * 100:.2f}%")

    # 显示最高概率的类别
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_name = class_names.get(predicted_class, f"未知类别_{predicted_class}")
    print(f"\n预测结果: {predicted_name}")
    print(f"置信度: {confidence * 100:.2f}%")

    return predictions

if __name__ == "__main__":

    model_path = "mobilenetv2_fruits.h5"
    image_path = os.getcwd() + "/test_image/" + "Apple.png"

    detect_image(image_path, model_path)
