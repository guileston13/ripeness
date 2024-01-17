##pip3 install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
import tensorflow as tf
import paho.mqtt.client as mqtt #expected imports
   
import RPi.GPIO as GPIO 
import time  
import cv2 
from PIL import Image
import numpy as np 
import smbus2
import os 
 

DEVICE_ADDRESS = 0x27

# Define commands
LCD_CHR = 1
LCD_CMD = 0 
LCD_LINE_1 = 0x80
LCD_LINE_2 = 0xC0
LCD_LINE_3 = 0x90
# Define delays
E_PULSE = 0.0005
E_DELAY = 0.0005

# Initialize I2C bus
bus = smbus2.SMBus(1)
#Image Counter
start_time = time.time()
img_counter = 0


BUTTON_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(-2) 


cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
cap1.set(cv2.CAP_PROP_FPS, 15)
cap2.set(cv2.CAP_PROP_FPS, 15)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# Color Detection 

def color_brown(img):   
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brown_lower = np.array([6, 63, 0], dtype=np.uint8)
        brown_upper = np.array([23, 255, 81], dtype=np.uint8)
        mask = cv2.inRange(hsv, brown_lower, brown_upper)
        #output = cv2.bitwise_and(img, img, mask=mask)
        brown = float(mask.sum()) / float(max_value)
        print('brown pixel percentage:', np.round(brown*100, 2))
        return brown

def color_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower =np.array([25, 52, 72],dtype=np.uint8)
    green_upper =np.array([102, 255, 255],dtype=np.uint8)

    mask = cv2.inRange(hsv, green_lower, green_upper)
    
   # output = cv2.bitwise_and(img, img, mask=mask)

    green = float(mask.sum()) / float(max_value) 

    print('green pixel percentage:', np.round(green*100, 2)) 
    return green

def color_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([22,60,200],dtype=np.uint8)
    yellow_upper = np.array([60,255,255],dtype=np.uint8)

    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

   # output = cv2.bitwise_and(img, img, mask=mask)
   
    yellow = float(mask.sum()) / float(max_value)

    print('yellow pixel percentage:', np.round(yellow*100, 2))
    return yellow

def largest(arr, n):
    max = arr[0]
    for i in range(1, n):
        if arr[i] > max:
            max = arr[i] 
    return max

# Define image processing function
def classify_ripeness(image):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="/home/pi/Desktop/fruit_ripeness_model_600x600.tflite")#path sa model
    interpreter.allocate_tensors()

    # Get the input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image = cv2.resize(image,(600,600))
    # Resize the image para sa model
    image = np.array(image, dtype=np.float32) / 255.0 # PIXEL VALUES NI SIYA
    image = np.expand_dims(image, axis=0) # Add a batch dimension huehue
    
   
    # Perform inference on the image
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class 
    predicted_class = np.argmax(output[0])
    class_labels = ['Overripe', 'Ripe', 'Unripe']
    class_name = class_labels[predicted_class]
    print('Prediction Output:', class_name)
    # Return the predicted class
    return class_name
    

# Display classification result on OLED LC
# Initialize display
def lcd_init():
    # Initialise display sa LCD PART!!!
    lcd_byte(0x33,LCD_CMD) # 110011 Initialise
    lcd_byte(0x32,LCD_CMD) # 110010 Initialise
    lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
    lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off
    lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
    lcd_byte(0x01,LCD_CMD) # 000001 Clear display
    time.sleep(E_DELAY)
def lcd_byte(bits, mode):
    bits_high = mode | (bits & 0xF0) | 0x08
    bits_low = mode | ((bits<<4) & 0xF0) | 0x08
    # High bits
    bus.write_byte(DEVICE_ADDRESS, bits_high)
    lcd_toggle_enable(bits_high)
    # Low bits
    bus.write_byte(DEVICE_ADDRESS, bits_low)
    lcd_toggle_enable(bits_low)
def lcd_toggle_enable(bits):
    # Toggle enable
    time.sleep(E_DELAY)
    bus.write_byte(DEVICE_ADDRESS, (bits | 0x04))
    time.sleep(E_PULSE)
    bus.write_byte(DEVICE_ADDRESS,(bits & ~0x04))
    time.sleep(E_DELAY)
def lcd_string(message,line):
    # Send string to display
    message = message.ljust(20," ")
    lcd_byte(line, LCD_CMD)
    for i in range(16):
        lcd_byte(ord(message[i]),LCD_CHR)
    
#MQTT PART
def on_connect():
    broker_address = "169.254.230.81"
    broker_port = 1884

    # Define the topic to publish to
    topic = "test"

    # Define the message to publish
    message = "hello"
    # Define the MQTT client
    client = mqtt.Client()
    # Connect to the MQTT broker 
    client.connect(broker_address, broker_port)
    # Publish the message to the topic
    client.publish(topic, message)

def merge_picture():
            
            img_01 = Image.open("/home/pi/images/1.jpg")
            img_02 = Image.open("/home/pi/images/2.jpeg")
            img_03 = Image.open("/home/pi/images/3.jpg")
            img_04 = Image.open("/home/pi/images/4.jpeg")


            img_01_size = img_01.size
            img_02_size = img_02.size
            img_03_size = img_02.size
            img_02_size = img_02.size
            print('img 1 size: ', img_01_size)
            print('img 2 size: ', img_02_size)
            print('img 3 size: ', img_03_size)
            print('img 4 size: ', img_03_size)
            new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
            new_im.paste(img_01, (0,0))
            new_im.paste(img_02, (img_01_size[0],0))
            new_im.paste(img_03, (0,img_01_size[1]))
            new_im.paste(img_04, (img_01_size[0],img_01_size[1]))                    # Sa

            new_im_resized = new_im.resize((600,600))
            new_im_resized.save("/home/pi/images/merged_images.png","PNG")
 
while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        cv2.imshow("camera1",frame1) 
        cv2.imshow("camera2",frame2)#Walaon ni later

     # Wait for button press
        if GPIO.input(BUTTON_PIN) == GPIO.LOW:
            print("detected")
            on_connect()
            time.sleep(5); #code sleep 5 seconds
            curr_time=int(time.time())
            time_elasped=curr_time-int(start_time)
            
            if(time_elasped==1):
                break
            #sa image pag save ni diri
            img_name = "1.jpg"
            img_name2 = "2.jpeg"
            cv2.imwrite('/home/pi/images/'+img_name, cv2.resize(frame1,(300,300)))
            cv2.imwrite('/home/pi/images/'+img_name2, cv2.resize(frame2,(300,300)))
            
            print("{} written!".format(img_name))
            print("{} written!".format(img_name2))
            img_counter += 1
            time.sleep(0.1)
            merge_picture()
            if(img_counter == 1): 
                cap1.release()
                cap2.release() 
                GPIO.cleanup()
                cv2.destroyAllWindows()      
                break
        if cv2.waitKey(1) == ord("q"):
            break
img = cv2.imread('/home/pi/images/merged_images.png') #image path for the merged
src_height, src_width, src_channels = img.shape #calculating the image properties 
max_value = src_height * src_width * 255

# Process image and classify ripeness
ripeness = classify_ripeness(img)
brown = color_brown(img)
green = color_green(img)
yellow = color_yellow(img)

 
output = [brown,green,yellow]
n = len(output)
Ans = largest(output, n)

print("Largest pixel is ", np.round(Ans*100,2))

if Ans == brown:
    print("Ripe") 
elif Ans == green:
    print("unripe")
elif Ans == yellow:
    print("Overripe")
else:
    print("liar")

    
print(ripeness)

if Ans == brown and ripeness == 'Ripe':
    finalOutput = "Ripe"
    lcd_init()
    lcd_string("    " + finalOutput, LCD_LINE_1)
    
elif Ans == green and ripeness == 'Unripe':
    finalOutput = "Unripe"
    lcd_init()
    lcd_string("    " + finalOutput, LCD_LINE_1)

elif Ans == yellow and ripeness == 'Overripe':
    finalOutput = "Overripe"
    lcd_init()
    lcd_string("    " + finalOutput, LCD_LINE_1)

else:
    finalOutput = "Outliar"
    lcd_init() 
    lcd_string("    " + finalOutput, LCD_LINE_1)
