#Harris Corner Detection

import cv2
import numpy as np
import random

imgs = ['exemplo1.jpg', 'exemplo2.jpg', 'exemplo3.jpg']

def harris(imagem, blockSize, ksize, k):
    # Carregar a imagem em escala de cinza

    # Aplicar o detector de Harris
    harris = cv2.cornerHarris(imagem, blockSize, ksize, k)
    harris = cv2.dilate(harris, None)

    # Destacar os cantos na imagem original
    imagem[harris > 0.01 * harris.max()] = [255]

    # Exibir resultados
    cv2.imshow('Harris Corner', imagem)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exec_harris():
    bSize_list = [3, 5, 7]  # Tamanhos de bloco possíveis
    ksize_list = [3, 5, 7]  # Tamanhos de kernel possíveis
    k_list = [0.04, 0.06, 0.08]  # Valores de k possíveis

    for img in imgs:
        # Ler a imagem em escala de cinza
        imagem = cv2.imread(f'uploads/{img}', cv2.IMREAD_GRAYSCALE)
        for i in range(3):
            blockSize = bSize_list[i]  # Tamanho do bloco
            ksize = ksize_list[i]  # Tamanho do kernel (deve ser ímpar)
            k = k_list[i]
            # Aplicar o detector de Harris
            harris(imagem.copy(), blockSize, ksize, k)

def shi_tomasi():
    maxCorners_size = [100, 200, 300]  # Número máximo de cantos a serem detectados
    qualityLevel_size = [0.01, 0.02, 0.03]  # Nível de qualidade para a detecção de cantos
    minDistance_size = [10, 20, 30]  # Distância mínima entre os cantos detectados
    # Aplicar o detector Shi-Tomasi
    for img in imgs:
        for i in range(3):
            # Ler a imagem em escala de cinza
            maxCorners = maxCorners_size[i]
            qualityLevel = qualityLevel_size[i]
            minDistance = minDistance_size[i]
            imagem = cv2.imread(f'uploads/{img}', cv2.IMREAD_GRAYSCALE)
            cantos = cv2.goodFeaturesToTrack(imagem, maxCorners, qualityLevel, minDistance)
            cantos = np.int64(cantos)
            # Marcar os cantos na imagem
            for canto in cantos:
                x, y = canto.ravel()
                cv2.circle(imagem, (x, y), 3, 255, -1)

            # Exibir resultados
            cv2.imshow('Shi-Tomasi', imagem)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



exec_harris()
shi_tomasi()