import cv2
import numpy as np
import random
imgs = ['exemplo1.jpg', 'exemplo2.jpg', 'exemplo3.jpg']

def sift():
    for img in imgs:
        # Criar o objeto SIFT
        sift = cv2.SIFT_create()

        # Detectar e computar características
        imagem = cv2.imread(f'uploads/{img}', cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(imagem, None)

        # Desenhar keypoints na imagem
        imagem_sift = cv2.drawKeypoints(imagem, keypoints, None)

        # Exibir resultados
        cv2.imshow('SIFT', imagem_sift)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

sift()