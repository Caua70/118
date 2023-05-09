import cv2


video = cv2.VideoCapture("haarcascade_fullbody.mxl")

img = video.read()

bbox = cv2.selectROI("tracking",img,False)


# Crie nosso classificador de corpos
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.mxl')


# Inicie a captura de vídeo para o arquivo de vídeo
cap = cv2.VideoCapture('walking.avi')

# Faça o loop assim que o vídeo for carregado com sucesso
while True:
    
    # Leia o primeiro quadro
    ret, frame = cap.read()

    # Converta cada quadro em escala de cinza
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    
    # Passe o quadro para nosso classificador de corpos
    bodies = body_classifier.detectMultiScale( 1.2, 3)
    
    
    # Extraia as caixas delimitadoras para quaisquer corpos identificados
    x,y,w,h = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x+w),(y+h)),(255,0,255),3,1)
    cv2.imshow("resultado", img)
    

    if cv2.waitKey(1) == 32: #32 é a barra de espaço
        break

cap.release()
cv2.destroyAllWindows()
