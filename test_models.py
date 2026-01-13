import pytest
import torch
from PIL import Image
import requests
from io import BytesIO
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForVision2Seq
import torchreid


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_image():
    url = "https://ultralytics.com/images/bus.jpg"
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def test_yolo(device, test_image):
    model = YOLO("yolov8n.pt").to(device)
    results = model(test_image, verbose=False)
    assert len(results[0].boxes) > 0
    print(f"\nYOLO detected {len(results[0].boxes)} objects")


def test_llava(device, test_image):
    processor = AutoProcessor.from_pretrained(
        "llava-v1.6-mistral-7b-hf",
        low_cpu_mem_usage=True
    )
    model = AutoModelForVision2Seq.from_pretrained(
        "llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    inputs = processor(text="What do you see?", images=test_image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)
    result = processor.decode(output[0], skip_special_tokens=True)
    assert len(result) > 10
    print(f"\nLLaVA response: {result}")


def test_osnet(device, test_image):
    model = torchreid.models.build_model(
        name="osnet_x1_0_imagenet.pt",
        num_classes=1000,
        pretrained=True
    ).to(device).eval()

    transform = torchreid.data.transforms.build_transforms()[0]
    input_tensor = transform(test_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(input_tensor)
    assert features.shape[1] == 512
    print(f"\nOSNet features shape: {features.shape}")