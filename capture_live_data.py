
import csv
import time
from pathlib import Path

import numpy as np
import serial

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 115200
MESSAGES_PER_CONDITION = 10

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "live_training_data"
OUTPUT_DIR.mkdir(exist_ok=True)

MESSAGES = [
    "HELLO WORLD",
    "FIBER ML TEST 123",
    "MEET ME AT THE PARK",
    "OPTICAL FIBER",
    "ANN SIGNAL RECOVERY",
    "BENT CABLE TEST",
    "DATA TRANSMISSION",
    "LIGHT SIGNAL",
    "HELLO AGAIN",
    "FIBER DECODER",
]


def capture_message(arduino, message):
    payload = message.encode("utf-8")
    expected_bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    ).astype(int).tolist()

    arduino.reset_input_buffer()
    arduino.write(f"MSG:{payload.hex().upper()}\n".encode("ascii"))
    arduino.flush()

    readings = []
    started = False
    deadline = time.monotonic() + len(expected_bits) * 0.05 + 15

    while time.monotonic() < deadline:
        raw = arduino.readline()
        if not raw:
            continue

        line = raw.decode("utf-8", errors="replace").strip()

        if line.startswith("ERROR,"):
            raise RuntimeError(f"Arduino error: {line}")

        if line.startswith("START,"):
            announced = int(line.split(",")[1])
            if announced != len(expected_bits):
                raise RuntimeError(
                    f"Expected {len(expected_bits)} bits, "
                    f"Arduino announced {announced}"
                )
            started = True
            continue

        if line == "DONE":
            if not started:
                raise RuntimeError("Received DONE without START")
            if len(readings) != len(expected_bits):
                raise RuntimeError(
                    f"Expected {len(expected_bits)} readings, "
                    f"received {len(readings)}"
                )
            return expected_bits, readings

        parts = line.split(",")
        if len(parts) == 2:
            try:
                index, reading = map(int, parts)
            except ValueError:
                continue

            if index != len(readings):
                raise RuntimeError(
                    f"Unexpected bit index {index}; "
                    f"expected {len(readings)}"
                )
            readings.append(reading)

    raise TimeoutError(
        f"Transmission timed out for message: {message!r}"
    )


def main():
    print("Choose the current fiber condition:")
    print("1 = straight")
    print("2 = bent")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in {"1", "2"}:
        print("Invalid choice. No data collected.")
        return

    condition = "straight" if choice == "1" else "bent"
    output_file = OUTPUT_DIR / f"{condition}_live.csv"

    print(f"\nCondition selected: {condition}")
    print(f"Output file: {output_file}")
    print("Keep the fiber condition unchanged throughout collection.")

    all_rows = []

    with serial.Serial(PORT, BAUD_RATE, timeout=3) as arduino:
        time.sleep(2)
        arduino.reset_input_buffer()

        for trial, message in enumerate(MESSAGES, start=1):
            print(f"Trial {trial}/{len(MESSAGES)}: {message!r}")

            bits, readings = capture_message(arduino, message)

            for index, (bit, reading) in enumerate(zip(bits, readings)):
                all_rows.append({
                    "condition": condition,
                    "trial": trial,
                    "message": message,
                    "bit_index": index,
                    "original_bit": bit,
                    "sensor_reading": reading,
                })

            print(f"  Captured {len(readings)} readings")
            time.sleep(1)

    columns = [
        "condition", "trial", "message",
        "bit_index", "original_bit", "sensor_reading"
    ]

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} labeled samples to:")
    print(output_file)
    print("Collection complete.")


if __name__ == "__main__":
    main()
