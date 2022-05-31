char incoming_data[3];//CMD,data
char ipt;
int n = 0;

void setup(void)
{
  Serial.begin(9600);
  Serial.print("Arduino example\r\n");

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(6, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(13, OUTPUT);
}

void loop(void)
{
  while(Serial.available()){
    ipt = char(Serial.read());
    //store ipt if not eot
    if(ipt!='\n'){
        if(n>2){
            n=0;
            
            continue;    
        }
        incoming_data[n]=ipt;
        n++;
    }else{
        n=0;    
        Serial.print("received\r\n");
        switch(incoming_data[0]){
            case 'N':
                Serial.print("Arduino example\r\n");
                Serial.print("done\r\n");
                break;
            case 'S':
                digitalWrite(11, incoming_data[1]-'0'); //LOW if 0 received, HIGH if 1 received
                digitalWrite(13, incoming_data[1]-'0');
                Serial.print("done\r\n");
                break;
                
            case 'T':
                if(incoming_data[1] - '0' == 1){
                  analogWrite(6, 30); //Lower = 20 //Current set 43  //Max 50
                  digitalWrite(13, 1);
                  Serial.print("done\r\n");
                  break;
                  }
                else{
                    analogWrite(6, 0);
                    digitalWrite(13, 0);
                    Serial.print("done\r\n");
                    break;
                    }
                digitalWrite(LED_BUILTIN, incoming_data[1]-'0');
                Serial.print("done\r\n");
                break;
                        
            default:
                Serial.print("INVALID CMD\r\n");
                Serial.print("done\r\n");
                break; 
        }
    }
  }
  delay(1);
}
