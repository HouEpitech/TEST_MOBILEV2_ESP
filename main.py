import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import numpy as np
import kagglehub
import json
from datetime import datetime


# -- Setting -- #

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), 'dataset'))
DEVICE_MODE = 'CPU'
DATASET_NAME = "utkarshsaxenadn/fruits-classification"
DATASET_DIR = os.getcwd() + "/dataset/Fruits Classification"

def download_dataset(destination_dir):
    os.makedirs(destination_dir, exist_ok=True)
    metadata_file = os.path.join(destination_dir, "dataset_metadata.json")

    current_version = None
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                current_version = metadata.get('version')
        except:
                print("Erreur lors de la lecture des métadonnées existantes")

    try:
        os.environ['KAGGLE_DATA_DIR'] = destination_dir

        new_path = kagglehub.dataset_download(DATASET_NAME)

        if os.path.exists(new_path):
            import shutil
            if not new_path.startswith(destination_dir):
                for root, dirs, files in os.walk(new_path):
                    for file in files:
                        src = os.path.join(root, file)
                        rel_path = os.path.relpath(root, new_path)
                        dst_dir = os.path.join(destination_dir, rel_path)
                        os.makedirs(dst_dir, exist_ok=True)
                        dst = os.path.join(dst_dir, file)

                        # 如果目标文件存在，先删除
                        if os.path.exists(dst):
                            os.remove(dst)
                        # 移动文件
                        shutil.move(src, dst)

                new_path = destination_dir

        new_metadata = {
            'dataset_name': DATASET_NAME,
            'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'download_date': datetime.now().isoformat(),
            'path': new_path
        }

        with open(metadata_file, 'w') as f:
            json.dump(new_metadata, f, indent=4, ensure_ascii=False)

        print(f"L'ensemble de données a été téléchargé sur: {new_path}")
        print(f"Version: {new_metadata['version']}")

        return new_path

    except Exception as e:
        print(f"Erreur lors du téléchargement: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def verifier_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        print("L'ensemble de données est vide")
        return False

    if not os.path.isdir(dataset_path):
        print(f"Erreur : {dataset_path} n'est pas un dossier")
        return False

    if len(os.listdir(dataset_path)) == 0:
        print(f"Erreur : {dataset_path} est vide")
        return False

    print('Vérification réussie')
    return True

def verifier_gpu():
    print("TensorFlow Version : ", tf.__version__)

    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("\nCarte graphique non détectée")
        print("Devices actuellement disponibles : ")
        print(tf.config.list_physical_devices())
        return False

    print("\nGPU détectés:")
    for gpu in gpus:
        print(f"- {gpu}")

    print("\nCUDA Config : ")
    print("- CUDA valid : ", tf.test.is_built_with_cuda())
    print("- GPU valid : ", tf.test.is_built_with_gpu_support())

    print("\n VRAM Config : ")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
        print(" - GPU VRAM Valid")
    except RuntimeError as e:
        print(" - Erreur : Impossible de définir la croissance de la mémoire:", e)

    print('\nDes tests GPU:')

    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("- GPU Test de calcul réussi")
            print("- Résultats des tests:", c.numpy())
    except Exception as e:
        print("- GPU Échec du test:", str(e))

    return True

if __name__ == '__main__':
    download_dataset(DATA_DIR)

    if not verifier_dataset(DATA_DIR):
        exit()
    else:
        print('Ensemble de données chargé avec succès')

    if verifier_gpu():
        DEVICE_MODE = 'GPU'
        print("Mode de device : GPU")
    else:
        print("Mode de device : CPU")

    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)

    base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=50)

    model.save("mobilenetv2_fruits.h5")