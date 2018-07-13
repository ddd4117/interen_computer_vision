# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""
import datetime
import numpy as np
import tensorflow as tf
import cv2
import main as m
sess = None
imagePath1 = './res/test.png'
imagePath2 = './res/test2.png'
imagePath3 = './res/test3.png'  # 추론을 진행할 이미지 경로
modelFullPath = './res/output_graph.pb'  # 읽어들일 graph 파일 경로
labelsFullPath = './res/output_labels.txt'  # 읽어들일 labels 파일 경로
softmax_tensor_ = None
f = open(labelsFullPath, 'rb')
lines = f.readlines()
labels = [str(w).replace("\n", "") for w in lines]
def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(_list):
    answer = None
    # maybe insert float convertion here - see edit remark!

    for image_data in _list:
        start = datetime.datetime.now()
        predictions = sess.run(softmax_tensor_,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            # print('%s (score = %.5f)' % (human_string, score))
        answer = labels[top_k[0]]
        # print("[INFO] detection took: {}s\n".format(
        #        (datetime.datetime.now() - start).total_seconds()))
    return answer.replace("b","").replace("'","")[:-2]

if __name__ == '__main__':
    # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
    create_graph()
    myList = []
    myList.append(cv2.imencode('.png', cv2.imread("./res/test.png"))[1].tostring())
    myList.append(cv2.imencode('.png', cv2.imread("./res/test2.png"))[1].tostring())
    myList.append(cv2.imencode('.png', cv2.imread("./res/test3.png"))[1].tostring())
    run_inference_on_image(myList)
