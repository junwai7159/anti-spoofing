import cv2

def draw_bbox(img, x_center, y_center, w, h, color=(0, 255, 0)):
  x1 = int(x_center - w / 2)
  y1 = int(y_center - h / 2)
  x2 = int(x_center + w / 2)
  y2 = int(y_center + h / 2)

  cv2.rectangle(img, (x1, y1), (x2, y2), color, 10)