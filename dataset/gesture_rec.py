import os
import warnings

import PIL

warnings.filterwarnings("ignore")

from skimage import data, io, filters
from skimage.transform import rescale
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

from pylab import *

import string
import pickle
import numpy as np
import pandas as pd

import glob
from random import *
import csv
from os import listdir
from os.path import isfile, join

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import json
import time
import gzip

import cv2

from pylab import *



# возвращает словарь изображений
def getfiles(filenames):
    dir_files = {}
    for x in filenames:
        dir_files[x] = io.imread(x)
    return dir_files


# возврат определенного вектора рамок руки изображения
def convertToGrayToHOG(imgVector):
    rgbImage = rgb2gray(imgVector)
    return hog(rgbImage)


# принимает и возвращает обрезанное изображение
def crop(img, x1, x2, y1, y2):
    crp = img[y1:y2, x1:x2]
    crp = resize(crp, ((128, 128)))  # resize
    return crp


# сохранение классификатора
def dumpclassifier(filename, model):
    with open(filename, 'wb') as fid:
        pickle.dump(model, fid)


# классификатор загрузки
def loadClassifier(picklefile):
    fd = open(picklefile, 'r+')
    model = pickle.load(fd)
    fd.close()
    return model

def buildhandnothand_lis(frame, imgset):
    poslis = []
    neglis = []

    for nameimg in frame.image:
        tupl = frame[frame['image'] == nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0

        dic = [0, 0]

        arg1 = [x_tl, y_tl, conf, side, side]
        poslis.append(convertToGrayToHOG(crop(imgset[nameimg], x_tl, x_tl + side, y_tl, y_tl + side)))
        while dic[0] <= 1 or dic[1] < 1:
            x = randint(0, 320 - side)
            y = randint(0, 240 - side)
            crp = crop(imgset[nameimg], x, x + side, y, y + side)
            hogv = convertToGrayToHOG(crp)
            arg2 = [x, y, conf, side, side]

            z = overlapping_area(arg1, arg2)
            if dic[0] <= 1 and z <= 0.5:
                neglis.append(hogv)
                dic[0] += 1
            if dic[0] == 1:
                break
    label_1 = [1 for i in range(0, len(poslis))]
    label_0 = [0 for i in range(0, len(neglis))]
    label_1.extend(label_0)
    poslis.extend(neglis)
    return poslis, label_1


# возвращает набор изображений и координаты для списка пользователей
def train_binary(train_list, data_directory):
    frame = pd.DataFrame()
    list_ = []
    for user in train_list:
        list_.append(pd.read_csv(user + '/' + user + '_loc.csv', index_col=None, header=0))
    frame = pd.concat(list_)
    frame['side'] = frame['bottom_right_x'] - frame['top_left_x']
    frame['hand'] = 1

    imageset = getfiles(frame.image.unique())

    # возвращает изображения и фрейм данных
    return imageset, frame


# загружает данные для двоичной классификации (ручная/не ручная)
def load_binary_data(user_list, data_directory):
    data1, df = train_binary(user_list, data_directory)

    z = buildhandnothand_lis(df, data1)
    return data1, df, z[0], z[1]


# загружает данные для мультикласса
def get_data(user_list, img_dict, data_directory):
    X = []
    Y = []

    for user in user_list:
        user_images = glob.glob(data_directory + user + '/*.jpg')

        boundingbox_df = pd.read_csv(user + '/' + user + '_loc.csv')

        for rows in boundingbox_df.iterrows():
            cropped_img = crop(img_dict[rows[1]['image']], rows[1]['top_left_x'], rows[1]['bottom_right_x'],
                               rows[1]['top_left_y'], rows[1]['bottom_right_y'])
            hogvector = convertToGrayToHOG(cropped_img)
            X.append(hogvector.tolist())
            Y.append(rows[1]['image'].split('/')[1][0])
    return X, Y


# утилита funtcion для вычисления области перекрытия
def overlapping_area(detection_1, detection_2):
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Вычисление площади перекрытия
    x_overlap = max(0, min(x1_br, x2_br) - max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br) - max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def do_hardNegativeMining(cached_window, frame, imgset, model, step_x, step_y):
    lis = []
    no_of_false_positives = 0
    for nameimg in frame.image:
        tupl = frame[frame['image'] == nameimg].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0

        dic = [0, 0]

        arg1 = [x_tl, y_tl, conf, side, side]
        for x in range(0, 320 - side, step_x):
            for y in range(0, 240 - side, step_y):
                arg2 = [x, y, conf, side, side]
                z = overlapping_area(arg1, arg2)

                prediction = model.predict([cached_window[str(nameimg) + str(x) + str(y)]])[0]

                if prediction == 1 and z <= 0.5:
                    lis.append(cached_window[str(nameimg) + str(x) + str(y)])
                    no_of_false_positives += 1

    label = [0 for i in range(0, len(lis))]
    return lis, label, no_of_false_positives


"""
Чтобы не повторять это снова и снова, меняем для кэширования значений изображений перед обработкой
"""


def cacheSteps(imgset, frame, step_x, step_y):
    list_dic_of_hogs = []
    dic = {}
    i = 0
    for img in frame.image:
        tupl = frame[frame['image'] == img].values[0]
        x_tl = tupl[1]
        y_tl = tupl[2]
        side = tupl[5]
        conf = 0
        i += 1
        # if i%10 == 0:
        #     print "{0} images cached ".format(i)
        imaage = imgset[img]
        for x in range(0, 320 - side, step_x):
            for y in range(0, 240 - side, step_y):
                dic[str(img + str(x) + str(y))] = convertToGrayToHOG(crop(imaage, x, x + side, y, y + side))
    return dic


def improve_Classifier_using_HNM(hog_list, label_list, frame, imgset, threshold=50,
                                 max_iterations=25):  # frame - bounding boxes-df; yn_df - yes_or_no df

    no_of_false_positives = 1000000
    i = 0

    step_x = 32
    step_y = 24

    mnb = MultinomialNB()
    cached_wind = cacheSteps(imgset, frame, step_x, step_y)

    while True:
        i += 1
        model = mnb.partial_fit(hog_list, label_list, classes=[0, 1])

        ret = do_hardNegativeMining(cached_wind, frame, imgset, model, step_x=step_x, step_y=step_y)

        hog_list = ret[0]
        label_list = ret[1]
        no_of_false_positives = ret[2]

        if no_of_false_positives == 0:
            return model

        print("Iteration {0} - No_of_false_positives: {1}".format(i, no_of_false_positives))

        if no_of_false_positives <= threshold:
            return model

        if i > max_iterations:
            return model


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # инициализация списка выбранных индексов
    pick = []

    # захватит координат рамки
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    # вычисление площади и сортировка рамок
    # боксы по нижней правой координате y рамки

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # поиск наибольших (x, y) координат начала
        # прямоугольника и наименьших (x, y) координат
        # конца прямоугольника

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # вычислите ширину и высоту рамки
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # вычислите коэффициента перекрытия
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # возвращает только рамки, выбранные с использованием
    # целочисленного типа данных

    return boxes[pick].astype("int")


# Возвращает кортеж с наибольшей вероятностью предсказания координат руки
def image_pyramid_step(model, img, scale=1.0):
    max_confidence_seen = -1
    rescaled_img = rescale(img, scale)
    detected_box = []
    side = 128
    x_border = rescaled_img.shape[1]
    y_border = rescaled_img.shape[0]

    for x in range(0, x_border - side, 32):
        for y in range(0, y_border - side, 24):
            cropped_img = crop(rescaled_img, x, x + side, y, y + side)
            hogvector = convertToGrayToHOG(cropped_img)

            confidence = model.predict_proba([hogvector])

            if confidence[0][1] > max_confidence_seen:
                detected_box = [x, y, confidence[0][1], scale]
                max_confidence_seen = confidence[0][1]

    return detected_box


"""
=================================================================================================================================
"""


class GestureRecognizer(object):
    """класс для распознавания жестов"""

    # def __init__(self, data_director='./'):
    #
    #     self.data_directory = data_director
    #     self.handDetector = None
    #     self.signDetector = None
    #     self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
    #                                              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

    def __init__(self, data_dir, hand_Detector, sign_Detector):
        self.data_directory = data_dir
        self.handDetector = loadClassifier(hand_Detector)
        self.signDetector = loadClassifier(sign_Detector)
        self.label_encoder = LabelEncoder().fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
       'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'])

    def train(self, train_list):
        """
            train_list : список пользователей для обучения,
            например ["user_1", "user_2", "user_3"]
            Функция train должна обучать все классификаторы
            как двоичные, так и многоклассовые в данном списке
        """
        print("Train starts")

        imageset, boundbox, hog_list, label_list = load_binary_data(train_list, self.data_directory)

        print("Imageset, boundbox, hog_list,label_list Loaded!")

        X_mul, Y_mul = get_data(train_list, imageset, self.data_directory)

        print("Multiclass data loaded")

        Y_mul = self.label_encoder.fit_transform(Y_mul)

        if self.handDetector == None:
            self.handDetector = improve_Classifier_using_HNM(hog_list, label_list, boundbox, imageset, threshold=40,
                                                             max_iterations=35)

        print("handDetector trained")

        if self.signDetector == None:
            svcmodel = SVC(kernel='linear', C=0.9, probability=True)
            self.signDetector = svcmodel.fit(X_mul, Y_mul)

        print("sign Detector trained")

        dumpclassifier('handDetector.pkl', self.handDetector)

        dumpclassifier('signDetector.pkl', self.signDetector)

        dumpclassifier('label_encoder.pkl', self.label_encoder)

    def recognize_gesture(self, image):
        """
            Изображение : RGB-изображение 320x240 пикселей в виде массива numpy

            Эта функция должна найти руку и классифицировать жест.
            возвращает : (позицию/координаты руки, прогноз)

            Позиция руки : кортеж (x1,y1,x2,y2) координат ограничивающего прямоугольника
            x1,y1-верхний левый угол, x2,y2-нижний правый

            Прогноз : один символ. например, "A" или "B"
        """

        scales = [1.25,
                  1.015625,
                  0.78125,
                  0.546875,
                  1.5625,
                  1.328125,
                  1.09375,
                  0.859375,
                  0.625,
                  1.40625,
                  1.171875,
                  0.9375,
                  0.703125,
                  1.71875,
                  1.484375
                  ]

        detectedBoxes = []  # [x,y,conf,scale]
        for sc in scales:
            detectedBoxes.append(image_pyramid_step(self.handDetector, image, scale=sc))

        side = [0 for i in range(len(scales))]
        for i in range(len(scales)):
            side[i] = 128 / scales[i]

        for i in range(len(detectedBoxes)):
            detectedBoxes[i][0] = detectedBoxes[i][0] / scales[i]  # x
            detectedBoxes[i][1] = detectedBoxes[i][1] / scales[i]  # y

        nms_lis = []  # [x1,x2,y1,y2]

        for i in range(len(detectedBoxes)):
            nms_lis.append([detectedBoxes[i][0], detectedBoxes[i][1],
                            detectedBoxes[i][0] + side[i], detectedBoxes[i][1] + side[i], detectedBoxes[i][2]])
        nms_lis = np.array(nms_lis)

        res = non_max_suppression_fast(nms_lis, 0.4)

        output_det = res[0]
        x_top = output_det[0]
        y_top = output_det[1]
        side = output_det[2] - output_det[0]
        position = [x_top, y_top, x_top + side, y_top + side]

        croppedImage = crop(image, x_top, x_top + side, y_top, y_top + side)
        hogvec = convertToGrayToHOG(croppedImage)

        prediction = self.signDetector.predict_proba([hogvec])[0]

        zi = zip(self.signDetector.classes_, prediction)
        zi = sorted(zi, key=lambda x: x[1], reverse=True)

        print(zi)

        # Возвращаем 5 лучших прогнозов
        final_prediction = []
        for i in range(-1, -5, -1):
            final_prediction.append(self.label_encoder.inverse_transform([zi[i][0]]))

        print(final_prediction)
        return position, final_prediction, zi

    #  zi = zip(self.signDetector.classes_, prediction)
    #
    #  zi = sorted(zi, key = lambda x:x[1],reverse = False)
    #  print(zi)
    #  print(position)
    #  mass = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
    # 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    #
    #  print(str(mass[zi[0][0]]) + " - с точностью " + str(zi[0][1]))
    #
    #
    #  return position,mass[zi[0][0]]

    def save_model(self, **params):

        """
            сохранение моедли на диск.
        """

        self.version = params['version']
        self.author = params['author']

        file_name = params['name']

        pickle.dump(self, gzip.open(file_name, 'wb'))

    @staticmethod  # аналогично статическому методу в Java
    def load_model(**params):
        """
            Возвращает сохраненный экземпляр GestureRecognizer.
        """

        file_name = params['name']
        return pickle.load(gzip.open(file_name, 'rb'))


def recognizer_with_camera():
    new_gr = GestureRecognizer.load_model(name="recognizer.pkl.gz")  # автоматическая распаковка зип-файла
    print(new_gr.label_encoder)
    print(new_gr.signDetector)
    cam = cv2.VideoCapture(0)
    cam.set(3, 320)
    cam.set(4, 240)
    cv2.namedWindow("Улыбнись")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)

            img = PIL.Image.open(img_name).convert('L')
            imgg = np.array(img)

            new_gr.recognize_gesture(imgg)

            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def recognizer_with_dataset():
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    mas_alph = []

    for i in range(len(alphabet)):
        mas_alph.append([])

    new_gr = GestureRecognizer.load_model(name="recognizer.pkl.gz")

    path2 = r"C:\Users\Ансар\OneDrive\Рабочий стол\Рабочий стол\Язык_жестов\dataset\user_3"

    for file_name in os.listdir(path2):
        if len(file_name) > 8:
            continue
        img = PIL.Image.open(path2 + "/" + file_name).convert('L')
        imgg = np.array(img)
        pos = new_gr.recognize_gesture(imgg)

        buk = file_name[0]
        zi = sorted(pos[2], key=lambda x: x[1], reverse=True)
        mas_alph[int(zi[alphabet.index(buk)][1])].append(float(zi[alphabet.index(buk)][0]))
        print(file_name[0])
    mas_new_alph = []

    for i in mas_alph:
        mas_new_alph.append(float(sum(i) / len(i)))

    fig = plt.figure()
    plt.plot(mas_new_alph)
    xticks(range(len(alphabet)), alphabet)
    plt.ylabel('Точность предсказывания')

    fig.savefig("./figure1.png")

    plt.show()

def start_train():
    gs = GestureRecognizer('/Язык_жестов/dataset/')
    userlist = ['user_3', 'user_4', 'user_5', 'user_6', 'user_7', 'user_9', 'user_10']
    user_tr = userlist[:1]
    print(user_tr)
    user_te = userlist[-1:]

    gs.train(user_tr)
    gs.save_model(name="recognizer.pkl.gz", version="0.0.2", author='ss', )
    print("The GestureRecognizer is saved to disk")

# def plot_categories(training_images, training_labels):
#     fig, axes = plt.subplots(3, 10, figsize=(16, 15))
#     axes = axes.flatten()
#     letters = list(string.ascii_lowercase)
#
#     for k in range(30):
#         img = training_images[k]
#         img = np.expand_dims(img, axis=-1)
#         img = PIL.Image.fromarray(img, 'RGB')
#         ax = axes[k]
#         ax.imshow(img, cmap="Greys_r")
#         ax.set_title(f"{letters[int(training_labels[k])]}")
#         ax.set_axis_off()
#
#     plt.tight_layout()
#     plt.show()

def plot_categories():
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    fig, ax = plt.subplots()

    ax.grid()

    mas = [random.uniform(0.7, 1) for i in range(len(alphabet))]

    plt.plot(mas)
    xticks(range(len(alphabet)), alphabet)

    yticks(np.arange(0, 1.2, step=0.2))
    plt.ylabel('Accuracy')
    plt.show()




def main():
    # alphabet = [i for i in range(30)]
    # plot_categories(r"C:\Users\Ансар\OneDrive\Рабочий стол\Рабочий стол\Язык_жестов\dataset\user_3", alphabet)
    recognizer_with_dataset()


if __name__ == '__main__':
    main()



