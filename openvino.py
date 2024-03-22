# import cv2 as cv

# print(cv.dnn.getAvailableBackends())

# net = cv.dnn.readNet('/yolov8_best_visdrone_openvino_model/yolov8_best_visdrone.xml',
#                      '/yolov8_best_visdrone_openvino_model/yolov8_best_visdrone.bin')

# cap = cv.VideoCapture(0)

# while cv.waitKey(1) < 0:
#     hasFrame, frame = cap.read()
#     if not hasFrame:
#         break

#     blob = cv.dnn.blobFromImage(frame, size=(672, 384))
#     net.setInput(blob)
#     out = net.forward()

#     for detection in out.reshape(-1, 7):
#         confidence = float(detection[2])
#         xmin = int(detection[3] * frame.shape[1])
#         ymin = int(detection[4] * frame.shape[0])
#         xmax = int(detection[5] * frame.shape[1])
#         ymax = int(detection[6] * frame.shape[0])

#         if confidence > 0.5:
#             cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

#     cv.imshow('OpenVINO face detection', frame)


import openvino as ov

core = ov.Core()
print(core.available_devices)
