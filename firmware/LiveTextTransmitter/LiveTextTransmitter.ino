#include <string.h>
const int laserPin = 8;
const int receiverPin = A0;
const int bitDelayMs = 50;

const int MAX_LINE_LENGTH = 200;
char inputLine[MAX_LINE_LENGTH];
int inputLength = 0;

int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

void setup() {
  pinMode(laserPin, OUTPUT);
  digitalWrite(laserPin, LOW);

  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\r') continue;

    if (c == '\n') {
      inputLine[inputLength] = '\0';
      processMessage(inputLine);
      inputLength = 0;
    } else if (inputLength < MAX_LINE_LENGTH - 1) {
      inputLine[inputLength++] = c;
    } else {
      inputLength = 0;
      Serial.println("ERROR,LINE_TOO_LONG");
    }
  }
}

void processMessage(char *line) {
  if (strncmp(line, "MSG:", 4) != 0) {
    Serial.println("ERROR,INVALID_COMMAND");
    return;
  }

  char *hex = line + 4;
  int hexLength = strlen(hex);

  if (hexLength == 0 || hexLength % 2 != 0) {
    Serial.println("ERROR,INVALID_HEX");
    return;
  }

  int byteCount = hexLength / 2;

  // Safety limit for Arduino UNO memory and serial handling.
  if (byteCount > 80) {
    Serial.println("ERROR,MESSAGE_TOO_LONG");
    return;
  }

  // Validate the complete message before transmitting.
  for (int i = 0; i < hexLength; i++) {
    if (hexValue(hex[i]) < 0) {
      Serial.println("ERROR,INVALID_HEX");
      return;
    }
  }

  Serial.print("START,");
  Serial.println(byteCount * 8);

  int bitIndex = 0;

  for (int i = 0; i < hexLength; i += 2) {
    int value = (hexValue(hex[i]) << 4) |
                 hexValue(hex[i + 1]);

    for (int bit = 7; bit >= 0; bit--) {
      int transmittedBit = (value >> bit) & 1;

      digitalWrite(
        laserPin,
        transmittedBit ? HIGH : LOW
      );

      delay(bitDelayMs);

      int reading = analogRead(receiverPin);

      // Return index and raw sensor reading.
      // Do not return the original bit as a decoder input.
      Serial.print(bitIndex++);
      Serial.print(",");
      Serial.println(reading);
    }
  }

  digitalWrite(laserPin, LOW);
  Serial.println("DONE");
}