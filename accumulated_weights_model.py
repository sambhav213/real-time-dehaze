import opencv-python
import math
import numpy as np
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
    indices = darkvec.argsort()        
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
    count = 0
    first_iter = True
    result = None
    while True:
        ret_val, frame = cap.read()
        if ret_val == True:
            if mirror:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (304,171))
            if first_iter:
                avg = np.float32(frame)
                first_iter = False
            cv2.accumulateWeighted(frame, avg, 0.01)
            result = cv2.convertScaleAbs(avg)
            count = count +1        
            if(count>=5):
                I = np.float32(result) / 255
                dark = DarkChannel(I,15)
                A = AtmLight(I,dark)
                te = TransmissionEstimate(I,A,15)
                tr = TransmissionRefine(frame,te)
                Reco = Recover(I,tr,A,0.1)
                cv2.namedWindow('Webcam2',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Webcam2', 800, 600)           
                cv2.imshow('Webcam2', Reco)
                cv2.imshow('Webcam1', frame)
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
