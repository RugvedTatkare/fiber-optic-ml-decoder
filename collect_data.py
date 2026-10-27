
import csv
import time
from pathlib import Path
import serial

PORT = "/dev/cu.usbmodem1101"
BAUD_RATE = 115200


folder = Path(__file__).parent / "final_data_enclosed"
folder.mkdir(exist_ok=True)

condition = input("Fiber condition (straight/bent): ").strip().lower()
trial = input("Trial number (e.g. 1): ").strip()

if condition not in ("straight", "bent"):
    raise ValueError("Condition must be straight or bent.")

filename = folder / f"{condition}_auto_trial_{int(trial):02d}.csv"

if filename.exists():
    raise FileExistsError(
        f"{filename.name} already exists. Choose another trial number."
    )

print(f"\nConnecting to Arduino on {PORT}...")

with serial.Serial(PORT, BAUD_RATE, timeout=3) as arduino:
    # Opening the port resets most Arduino Uno boards.
    time.sleep(2)
    arduino.reset_input_buffer()

    print("Starting collection...")
    arduino.write(b"S")

    with filename.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "bit_index",
            "original_bit",
            "sensor_reading",
            "threshold_bit",
            "is_error",
            "fiber_condition",
            "trial",
        ])

        rows_saved = 0

        while True:
            line = arduino.readline().decode(
                "utf-8", errors="replace"
            ).strip()

            if not line:
                continue

            if line == "TRANSMISSION_COMPLETE":
                break

            parts = line.split(",")

            # Ignore READY and the Arduino CSV header.
            if len(parts) != 5 or parts[0] == "bit_index":
                continue

            try:
                bit_index, original_bit, reading, decoded_bit, error = (
                    int(value) for value in parts
                )
            except ValueError:
                continue

            writer.writerow([
                bit_index,
                original_bit,
                reading,
                decoded_bit,
                error,
                condition,
                int(trial),
            ])

            rows_saved += 1

            if rows_saved % 100 == 0:
                print(f"Collected {rows_saved} bits...")

print(f"\nDone! Saved {rows_saved} rows to {filename.name}")
