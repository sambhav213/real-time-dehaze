import opencv-python
import numpy as np
import image_dehazer
def split_video_channels(mirror=False):
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam1',cv2.WINDOW_NORMAL)
    ch_input_c ="1"
    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            if mirror:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (304,171))
            height, width, layers = frame.shape
            HazeCorrectedImg, haze_map = image_dehazer.remove_haze(frame, showHazeTransmissionMap=False)
            (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
            B_sum = np.sum(Bo)
            G_sum = np.sum(Go)
            R_sum = np.sum(Ro)
            if (G_sum >= B_sum): 
                if (R_sum >= G_sum): 
                    if((R_sum/G_sum) >= 1.1): 
                        ch_input_c = "6" 
                    else:
                        ch_input_c = "1" 
                else:
                    if((G_sum/R_sum) >= 1.1): 
                        ch_input_c = "3" 
                    else:
                        ch_input_c = "1" 
            else:
                if (R_sum >= B_sum): 
                    if((R_sum/B_sum) >= 1.1): 
                        ch_input_c = "1" 
                    else:
                        ch_input_c = "5" 
                else:
                    if((B_sum/R_sum) >= 1.1):
                        ch_input_c = "3" 
                    else:
                        ch_input_c = "5" 
            if(ch_input_c == "1"):
                cv2.imshow('Webcam1', HazeCorrectedImg)
            elif(ch_input_c == "2"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Bo = cv2.merge([Bo, zeroImgMatrix, zeroImgMatrix])
                cv2.imshow('Webcam1', Bo)
            elif(ch_input_c == "3"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Go = cv2.merge([zeroImgMatrix, Go, zeroImgMatrix])
                cv2.imshow('Webcam1', Go)
            elif(ch_input_c == "4"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Ro = cv2.merge([zeroImgMatrix, zeroImgMatrix , Ro]) 
                cv2.imshow('Webcam1', Ro)
            elif(ch_input_c == "5"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Co = cv2.merge([Bo,Go , zeroImgMatrix]) 
                cv2.imshow('Webcam1', Co)
            elif(ch_input_c == "6"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Yo = cv2.merge([zeroImgMatrix, Go, Ro]) 
                cv2.imshow('Webcam1', Yo)
            elif(ch_input_c == "7"):
                (Bo, Go, Ro) = cv2.split(HazeCorrectedImg)
                zeroImgMatrix = np.zeros((height, width), dtype="uint8")
                Mo = cv2.merge([Bo, zeroImgMatrix, Ro]) 
                cv2.imshow('Webcam1', Mo)
            else:
                cv2.imshow('Webcam1', HazeCorrectedImg)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
def main():
    split_video_channels(mirror=True)
if __name__ == '__main__':
    main()
