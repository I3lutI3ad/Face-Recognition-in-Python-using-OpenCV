import cv2, os
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import time
import pyaudio
import wave

cap = cv2.VideoCapture(0)
cascadePath = "lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()

def play_sound(file_path):
    fs, data = wavfile.read(file_path)
    sd.play(data,fs)
def record_name(file_path):
    CHUNK = 1024
    FORMAT = pyaudio.paInt32
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = file_path

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not rread the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image = cv2.imread(image_path)  #Image.open(image_path).convert('L')
        image_pil = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

def train_recogniser():
    global subject
    # Path to the face Dataset
    path = 'faces'
    # Call the get_images_and_labels function and get the face images and the
    # corresponding labels
    images, labels = get_images_and_labels(path)
    cv2.destroyAllWindows()
    subject = labels[len(labels)-1]
    # Perform the tranining
    recognizer.train(images, np.array(labels))

train_recogniser()
flag =0
name_flag = 0
count=0
name_say=0
while(True):
    if(flag>1):
        time.sleep(2)
    ret, frame = cap.read()
    small = cv2.resize(frame,(160,120))
    #print flag
    if(flag>=1):
        if(flag == 6):
            flag=0
            train_recogniser()
            play_sound("Sounds/name.wav")
            time.sleep(3)
            play_sound("Sounds/beep.wav")
            record_name("Names/"+str(subject)+".wav")
            continue
        else:
            if(flag==1):
                play_sound("Sounds/new_face.wav")
                time.sleep(7)
                play_sound("Sounds/beep.wav")
                time.sleep(1)
            play_sound("Sounds/cam.wav")
            cv2.imwrite ("faces/"+str(subject+1)+"."+str(flag)+".png",small)
            flag=flag+1
    if(flag==0):
        predict_image_pil = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        predict_image = np.array(predict_image_pil, 'uint8')
        faces = faceCascade.detectMultiScale(predict_image)
        for (x, y, w, h) in faces:
            cv2.rectangle(small,(x,y),(x+w,y+w),(255,255,0),1)
            #print w
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
    #       nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            #if nbr_actual == nbr_predicted:
            #print str(conf) + ' ' +str(nbr_predicted)
            if(w>55 and conf<60):
                print str(w)+' '+str(conf) + ' ' +str(nbr_predicted)
                if(name_flag==nbr_predicted):
                    if(count>11):
                        count=0
                    count=count+1
                else:
                    name_flag=nbr_predicted
                    count=0
                    name_say=0
                if(count==10 and name_say==0):
                    play_sound("Sounds/hello.wav")
                    time.sleep(1)
                    play_sound("Names/"+str(nbr_predicted)+".wav")
                    time.sleep(2)
                    play_sound("Sounds/greetings.wav")
                    time.sleep(3)
                    name_say=1
                #print subject
            elif(w>55 and conf>80):
                flag = 1
                #play_sound("Sounds/smile.wav")
                #time.sleep(2)
                #for num in range(0,5):
 #                  play_sound("Sounds/cam.wav")
#                   cv2.imwrite (str(subject+1)+"."+str(num)+".png",small)
#                   time.sleep(0.5)
            cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
            cv2.waitKey(100)
    cv2.imshow('frame',small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
