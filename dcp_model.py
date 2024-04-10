import opencv-python
import math;
import numpy as np;
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = [line.strip() for line in open("coco.names")]
output_layers = net.getUnconnectedOutLayersNames()
class_colors = np.random.uniform(0, 255, size=(len(classes), 3))
def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x, y = center_x - w // 2, center_y - h // 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes, class_ids, confidences
def draw_boxes(image, indices, boxes, class_ids, confidences):
    for i in indices:
        i = i
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = class_colors[class_ids[i]]
        color = tuple(map(int, color))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)
    indices = darkvec.argsort();        
    indices = indices[imsz-numpx::]
    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
        atmsum = atmsum + imvec[indices[ind]]
    A = atmsum / numpx
    return A
def TransmissionEstimate(im,A,sz):
    omega = 1
    im3 = np.empty(im.shape,im.dtype)
    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]
    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission
def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I
    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
    q = mean_a*im + mean_b
    return q
def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)
    return t
def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)
    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    return res
def split_video_channels(mirror=False):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam1', 800, 600)
    ch_input_h ="0"
    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            if mirror:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (304,171))
            indices, boxes, class_ids, confidences = detect_objects(frame)
            draw_boxes(frame, indices, boxes, class_ids, confidences)
            I = np.float32(frame) / 255
            dark = DarkChannel(I,15)
            A = AtmLight(I,dark)
            te = TransmissionEstimate(I,A,15)
            tr = TransmissionRefine(frame,te)
            Reco = Recover(I,tr,A,0.1)
            if(ch_input_h == "i"):
                haze_map = tr / np.max(tr)
                haze_map = np.power(haze_map, 2.5)
                heatmap = cv2.applyColorMap(np.uint8(haze_map * 255), cv2.COLORMAP_JET)
                cv2.namedWindow('Webcam2_colormap_jet',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Webcam2_colormap_jet', 800, 600)
                cv2.imshow('Webcam2_colormap_jet', heatmap)
            elif(ch_input_h == "j"):
                cv2.destroyWindow
            else:
                cv2.imshow('Webcam1', Reco)  
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('i'):
            ch_input_h= "i"
        elif cv2.waitKey(1) & 0xFF == ord('j'):
            ch_input_h = "j"
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def main():
    split_video_channels(mirror=True)
if __name__ == '__main__':
    main()
