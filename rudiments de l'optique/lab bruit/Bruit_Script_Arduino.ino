/*
  Bruit: un script Arduino qui fait allumer une DEL avec un peu de bruit.
  Un signal de synchronisation (LOW/HIGH) delimite l'experience et le background.

*/

#define LED DAC1
#define SYNC 4

void setup() {
  pinMode(SYNC, OUTPUT);
}

void loop() {

  int i = 0;
  int randNumber = random(10);
  int signal = 10;
  int background = 220;

  digitalWrite(SYNC, LOW); 

  for (i=0; i < 100; i++) {
    int randNumber = random(10);
    analogWrite(LED, background + randNumber);
    delay(1);
  }

  digitalWrite(SYNC, HIGH);  

  for (i=0; i < 100; i++) {
    int randNumber = random(10);
    analogWrite(LED, background + signal + randNumber); 
    delay(1);
  }
}