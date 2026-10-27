# Fiber Optic ML Decoder

> Machine learning-based decoding of optical signals to recover transmitted bits and text messages under varying fiber conditions.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![Machine Learning](https://img.shields.io/badge/ML-Artificial%20Neural%20Network-purple)
![Status](https://img.shields.io/badge/Status-Experimental-yellow)

## What is Fiber Optic ML Decoder?

Bending or pinching a fiber-optic cable can affect the received light signal and make digital communication less reliable.

This project explores whether an **Artificial Neural Network (ANN)** can recover transmitted bits from received sensor readings. It compares neural-network decoding with a conventional threshold-based approach and reconstructs the original text message.

## System Architecture

```text
Text Message
     ↓
Arduino-Based Transmitter
     ↓
Fiber-Optic Cable
     ↓
Light Sensor / Receiver
     ↓
Signal Readings
     ↓
┌───────────────────┬───────────────────┐
│ Threshold Decoder │ ANN-Based Decoder │
└───────────────────┴───────────────────┘
             ↓
       Predicted Bits
             ↓
       Text Recovery
             ↓
       Web Interface
```

## Key Features

- **Live transmission** — Transmit text messages using an Arduino-based setup.
- **Signal data collection** — Record sensor readings under straight and bent fiber conditions.
- **Dual decoding approach** — Compare fixed-threshold decoding with ANN predictions.
- **Neural-network inference** — Use a trained Keras model to predict transmitted bits.
- **Message recovery** — Reconstruct text from the decoded bit sequence.
- **Web interface** — Interact with the decoding workflow through a FastAPI backend and frontend.
- **Experimental evaluation** — Test message recovery using collected and live sensor data.

## Machine Learning

| Component | Description |
|---|---|
| Model | Artificial Neural Network (ANN) |
| Framework | TensorFlow / Keras |
| Preprocessing | Scikit-learn StandardScaler |
| Production model | `ml/models/bit_decoder.keras` |
| Feature scaler | `ml/models/bit_scaler.joblib` |
| Baseline | Fixed threshold decoder (531) |

## Experimental Results

In one live test session, four messages were recovered with **zero bit errors across 720 transmitted bits** under the tested bent-fiber configuration. In a separate test with a changed bend configuration, 6 errors occurred across 192 bits.

These results are specific to the tested conditions and do not establish perfect decoding across all fiber configurations.

## Tech Stack

- **Hardware:** Arduino-compatible microcontroller, optical transmitter, fiber cable and light sensor
- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Processing:** Python, NumPy, Pandas
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Data:** CSV datasets and live sensor recordings

## Project Structure

```text
fiber-optic-ml-decoder/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── firmware/
│   └── LiveTextTransmitter/
├── ml/
│   └── models/
│       ├── bit_decoder.keras
│       └── bit_scaler.joblib
├── final_data_enclosed/
├── live_training_data/
├── research/
├── capture_live_data.py
├── collect_data.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RugvedTatkare/fiber-optic-ml-decoder.git
cd fiber-optic-ml-decoder
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the backend

```bash
python -m uvicorn backend.main:app --reload
```

Backend: `http://127.0.0.1:8000`

API documentation: `http://127.0.0.1:8000/docs`

Health check: `http://127.0.0.1:8000/health`

Open `frontend/index.html` in a browser to access the interface. Hardware-based experiments additionally require the transmitter and receiver setup to be connected and configured.

## Research & Experiments

The `research/` directory contains the experimental notebook, candidate-model training script, and live message-recovery evaluation script.

The current production model is preserved separately from experimental candidate models. Retraining is not required to run the existing backend.

## Limitations

- Decoding performance depends on the physical fiber configuration and received signal quality.
- A fixed threshold may not generalize to changing bend conditions.
- Further testing is needed across more cable configurations and repeated trials.

## Future Work

- Evaluate decoding across different bend severities.
- Improve robustness to changes in received signal intensity.
- Compare bit error rate and message-recovery performance across methods.
- Expand live data collection and experimental validation.

---

**Developed as an applied machine-learning project exploring optical signal decoding and message recovery.**
