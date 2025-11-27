import os
import requests

API_URL = "http://0.0.0.0:8000/predict"

# updated path
TEST_DIR = os.path.join(os.path.dirname(__file__), "asl_alphabet_test")


def expected_label(filename: str):
    return filename.split("_")[0]


def test_all_asl_images():
    image_files = sorted([
        f for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    assert len(image_files) > 0, "No test images found."

    correct = 0
    total = 0

    for fname in image_files:
        total += 1
        expected = expected_label(fname)
        path = os.path.join(TEST_DIR, fname)

        with open(path, "rb") as f:
            response = requests.post(
                API_URL,
                files={"file": (fname, f, "image/jpeg")}
            )

        assert response.status_code == 200, f"API error for file {fname}"

        pred = response.json().get("prediction", None)
        assert pred is not None, f"Missing prediction for {fname}"

        if pred == expected:
            correct += 1
        else:
            print(f"Mismatch: {fname} → expected {expected}, got {pred}")

    accuracy = correct / total
    print(f"Accuracy = {accuracy*100:.2f}% ({correct}/{total})")

    assert accuracy > 0.5, "Model accuracy below 50%"
