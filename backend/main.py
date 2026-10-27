from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import time
import serial

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "ml" / "models"

# Load the saved ANN and its training scaler
model = tf.keras.models.load_model(
    MODEL_DIR / "bit_decoder.keras"
)
scaler = joblib.load(MODEL_DIR / "bit_scaler.joblib")

app = FastAPI(
    title="Fiber Optic ML Decoder",
    description="Predict transmitted bits from received sensor readings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecodeRequest(BaseModel):
    sensor_readings: list[float] = Field(min_length=1)


@app.get("/")
def home():
    return {
        "message": "Fiber Optic ML Decoder API is running",
        "model": "TensorFlow ANN",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict-bits")
def predict_bits(request: DecodeRequest):
    readings = np.asarray(
        request.sensor_readings,
        dtype=np.float32,
    ).reshape(-1, 1)

    scaled_readings = scaler.transform(readings)

    probabilities = model.predict(
        scaled_readings,
        verbose=0,
    ).reshape(-1)

    predictions = (probabilities >= 0.5).astype(int)

    threshold_predictions = (readings.reshape(-1) > THRESHOLD).astype(int)

    return {
        "sensor_readings": readings.reshape(-1).tolist(),
        "ann_bits": predictions.tolist(),
        "ann_probabilities": probabilities.round(4).tolist(),
        "threshold_bits": threshold_predictions.tolist(),
    }
from fastapi import HTTPException


def bits_to_text(bits):
    if len(bits) % 8 != 0:
        raise ValueError("Bit sequence must contain complete 8-bit bytes.")

    values = []
    for i in range(0, len(bits), 8):
        byte_bits = "".join(str(int(b)) for b in bits[i:i + 8])
        values.append(int(byte_bits, 2))

    return bytes(values).decode("utf-8", errors="replace")


@app.post("/decode-text")
def decode_text(request: DecodeRequest):
    readings = np.asarray(
        request.sensor_readings,
        dtype=np.float32
    ).reshape(-1, 1)

    if len(readings) % 8 != 0:
        raise HTTPException(
            status_code=400,
            detail="Provide readings for a complete number of 8-bit bytes."
        )

    scaled = scaler.transform(readings)
    probabilities = model.predict(scaled, verbose=0).reshape(-1)

    ann_bits = (probabilities >= 0.5).astype(int)
    threshold_bits = (readings.reshape(-1) > THRESHOLD).astype(int)

    return {
        "ann_bits": ann_bits.tolist(),
        "ann_text": bits_to_text(ann_bits),
        "threshold_bits": threshold_bits.tolist(),
        "threshold_text": bits_to_text(threshold_bits),
    }
# Live hardware transmission and ML decoding

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 115200
THRESHOLD = 531


class LiveMessageRequest(BaseModel):
    message: str = Field(min_length=1)


@app.post("/transmit-decode")
def transmit_decode(request: LiveMessageRequest):
    # Encode arbitrary text as UTF-8 bytes.
    payload = request.message.encode("utf-8")

    # Arduino firmware currently supports up to 80 bytes.
    if len(payload) > 80:
        raise HTTPException(
            status_code=400,
            detail="Message is too long. Maximum is 80 UTF-8 bytes."
        )

    # Convert message bytes to hexadecimal for serial transmission.
    hex_message = payload.hex().upper()
    expected_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    ).astype(int)

    readings = []
    completed = False

    try:
        # Serial Monitor must be closed so this API can use the port.
        with serial.Serial(PORT, BAUD_RATE, timeout=3) as arduino:
            # Opening serial usually resets an Arduino UNO.
            time.sleep(2)
            arduino.reset_input_buffer()

            arduino.write(f"MSG:{hex_message}\n".encode("ascii"))
            arduino.flush()

            deadline = time.monotonic() + max(
                10, len(expected_bits) * 0.05 + 10
            )

            while time.monotonic() < deadline:
                raw_line = arduino.readline()

                if not raw_line:
                    continue

                line = raw_line.decode(
                    "utf-8", errors="replace"
                ).strip()

                if not line or line == "READY":
                    continue

                if line.startswith("ERROR,"):
                    raise HTTPException(
                        status_code=502,
                        detail=f"Arduino reported: {line}"
                    )

                if line.startswith("START,"):
                    continue

                if line == "DONE":
                    completed = True
                    break

                # Arduino returns: bit_index,sensor_reading
                parts = line.split(",")

                if len(parts) == 2:
                    try:
                        bit_index = int(parts[0])
                        reading = int(parts[1])
                    except ValueError:
                        continue

                    if bit_index != len(readings):
                        raise HTTPException(
                            status_code=502,
                            detail="Received an unexpected bit index."
                        )

                    readings.append(reading)

    except serial.SerialException as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Could not communicate with the Arduino. "
                "Close Serial Monitor/Plotter and check the USB port. "
                f"Details: {exc}"
            )
        ) from exc

    if not completed or len(readings) != len(expected_bits):
        raise HTTPException(
            status_code=504,
            detail=(
                f"Incomplete transmission: received {len(readings)} "
                f"of {len(expected_bits)} readings."
            )
        )

    # Decode the actual received readings with the trained ANN.
    sensor_array = np.asarray(
        readings, dtype=np.float32
    ).reshape(-1, 1)

    scaled_readings = scaler.transform(sensor_array)

    probabilities = model.predict(
        scaled_readings, verbose=0
    ).reshape(-1)

    ann_bits = (probabilities >= 0.5).astype(int)
    threshold_bits = (sensor_array.reshape(-1) > THRESHOLD).astype(int)

    # Reconstruct text from each decoder's predicted bits.
    ann_text = bits_to_text(ann_bits)
    threshold_text = bits_to_text(threshold_bits)

    # Compare each prediction with the known transmitted message.
    ann_errors = int(np.sum(ann_bits != expected_bits))
    threshold_errors = int(
        np.sum(threshold_bits != expected_bits)
    )
    total_bits = len(expected_bits)
    ann_error_details = [
    {
        "bit_index": int(i),
        "expected_bit": int(expected_bits[i]),
        "adc_reading": int(readings[i]),
        "ann_probability": round(float(probabilities[i]), 4),
        "ann_prediction": int(ann_bits[i]),
        "threshold_prediction": int(threshold_bits[i]),
    }
    for i in np.where(ann_bits != expected_bits)[0]
    ]   
    # Signal diagnostics, grouped by the bit we transmitted.
    # These labels are used for analysis only, not as ANN inputs.
    sensor_values = np.asarray(readings, dtype=np.float32)

    zero_mask = expected_bits == 0
    one_mask = expected_bits == 1

    zero_readings = sensor_values[zero_mask]
    one_readings = sensor_values[one_mask]

    signal_diagnostics = {
        "bit_0_count": int(len(zero_readings)),
        "bit_0_mean": round(float(np.mean(zero_readings)), 2),
        "bit_0_min": int(np.min(zero_readings)),
        "bit_0_max": int(np.max(zero_readings)),
        "bit_1_count": int(len(one_readings)),
        "bit_1_mean": round(float(np.mean(one_readings)), 2),
        "bit_1_min": int(np.min(one_readings)),
        "bit_1_max": int(np.max(one_readings)),
        "mean_signal_gap": round(
            float(np.mean(one_readings) - np.mean(zero_readings)), 2
        ),
    }

    return {
        "original_message": request.message,
        "ann_decoded_message": ann_text,
        "threshold_decoded_message": threshold_text,
        "total_bits": total_bits,
        "sensor_readings": readings,
        "ann_bits": ann_bits.tolist(),
        "threshold_bits": threshold_bits.tolist(),
        "ann_bit_errors": ann_errors,
        "threshold_bit_errors": threshold_errors,
        "signal_diagnostics": signal_diagnostics,
        "ann_ber_percent": round(
            ann_errors / total_bits * 100, 3
        ),
        "threshold_ber_percent": round(
            threshold_errors / total_bits * 100, 3
        ),
        "ann_probabilities": probabilities.round(4).tolist(),
        "ann_error_details": ann_error_details,
    }