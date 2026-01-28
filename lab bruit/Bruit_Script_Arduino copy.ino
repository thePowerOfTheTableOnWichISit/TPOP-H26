#define LED DAC1
#define SYNC 4
#define DET A0   // entrée analogique du photodétecteur

void setup() {
  pinMode(SYNC, OUTPUT);
  pinMode(DET, INPUT);

  Serial.begin(115200); 
}

void loop() {

  int signal = 10;
  int background = 220;
  digitalWrite(SYNC, LOW);

  for (int i = 0; i < 100; i++) {
    int randNumber = random(10);
    int ledValue = background + randNumber;

    analogWrite(LED, ledValue);
    delay(0.2);

    int detValue = analogRead(DET);
    Serial.println(String(micros()) + "," + String(detValue));

    delay(0.8);
  }

  digitalWrite(SYNC, HIGH);
  for (int i = 0; i < 100; i++) {
    int randNumber = random(10);
    int ledValue = background + signal + randNumber;

    analogWrite(LED, ledValue);
    delay(0.2);

    int detValue = analogRead(DET);
    Serial.println(String(micros()) + "," + String(detValue));

    delay(0.8);
  }
}