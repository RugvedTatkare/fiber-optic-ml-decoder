
from pathlib import Path
import serial
import time
import binascii
import joblib
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
PORT = "/dev/cu.usbmodem1101"
BAUD = 115200
THRESHOLD = 531

MODEL_DIR = ROOT / "ml" / "models"
EXPERIMENT_DIR = MODEL_DIR / "live_experiment"

existing_model = tf.keras.models.load_model(
    MODEL_DIR / "bit_decoder.keras"
)
existing_scaler = joblib.load(
    MODEL_DIR / "bit_scaler.joblib"
)

candidate_model = tf.keras.models.load_model(
    EXPERIMENT_DIR / "candidate_decoder.keras"
)
candidate_scaler = joblib.load(
    EXPERIMENT_DIR / "candidate_scaler.joblib"
)


def text_to_bits(message):
    data = message.encode("utf-8")
    return "".join(f"{byte:08b}" for byte in data)


def bits_to_text(bits):
    usable = len(bits) - len(bits) % 8
    data = bytearray()
    for i in range(0, usable, 8):
        data.append(int(bits[i:i + 8], 2))
    return data.decode("utf-8", errors="replace")


def decode_bits(model, scaler, readings):
    X = np.array(readings, dtype=np.float32).reshape(-1, 1)
    X = scaler.transform(X)
    probabilities = model.predict(X, verbose=0).ravel()
    return "".join(
        "1" if p >= 0.5 else "0" for p in probabilities
    )


def collect_transmission(ser, message):
    payload = binascii.hexlify(
        message.encode("utf-8")
    ).decode("ascii")

    ser.reset_input_buffer()
    ser.write(f"MSG:{payload}\n".encode("ascii"))
    ser.flush()

    readings = []
    expected_count = None
    started = False
    deadline = time.time() + max(30, len(message) * 3)

    while time.time() < deadline:
        line = ser.readline().decode(
            "utf-8", errors="ignore"
        ).strip()

        if not line:
            continue

        if line.startswith("START,"):
            expected_count = int(line.split(",")[1])
            started = True
            readings = []
            print(f"Transmission started: {expected_count} bits")

        elif line == "DONE" and started:
            break

        elif started:
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    index = int(parts[0])
                    value = int(parts[1])
                    if index == len(readings):
                        readings.append(value)
                except ValueError:
                    pass

    if expected_count is None:
        raise RuntimeError(
            "No START response. Check the Arduino connection "
            "and confirm the live transmitter firmware is running."
        )

    if len(readings) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} readings but received "
            f"{len(readings)}. Please retry the transmission."
        )

    return readings


def report(name, predicted, expected):
    errors = sum(a != b for a, b in zip(predicted, expected))
    ber = errors / len(expected) * 100
    decoded = bits_to_text(predicted)

    print(f"\n{name}")
    print(f"Bit errors: {errors}/{len(expected)}")
    print(f"Bit error rate: {ber:.3f}%")
    print(f"Exact message recovery: {'YES' if predicted == expected else 'NO'}")
    print(f"Decoded text: {decoded!r}")


def main():
    message = input(
        "Enter a NEW test message using ordinary English text: "
    ).strip()

    if not message:
        print("Message cannot be empty.")
        return

    expected = text_to_bits(message)

    print("\nKeep the fiber in the BENT condition for this test.")
    input("Press Enter when the cable is ready...")

    try:
        with serial.Serial(PORT, BAUD, timeout=1) as ser:
            time.sleep(2)
            readings = collect_transmission(ser, message)

        print(f"\nOriginal message: {message!r}")
        print(f"Collected readings: {len(readings)}")
        print(f"Expected bits: {len(expected)}")

        existing_bits = decode_bits(
            existing_model, existing_scaler, readings
        )
        candidate_bits = decode_bits(
            candidate_model, candidate_scaler, readings
        )
        threshold_bits = "".join(
            "1" if value > THRESHOLD else "0"
            for value in readings
        )

        report("EXISTING ANN", existing_bits, expected)
        report("CANDIDATE ANN", candidate_bits, expected)
        report("THRESHOLD BASELINE (531)", threshold_bits, expected)

    except Exception as exc:
        print(f"\nTEST FAILED: {exc}")


if __name__ == "__main__":
    main()
