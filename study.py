import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


w0 = 50
w = w0
h = 30
scale = 0.95
test =[]
test = np.array(test)

def jiuzheng(edges, x, y):
    length = 5
    for i in range(length):
        for j in range(length):
            if edges[y-i][x-j] == 255:
                return x-j, y-i
            if edges[y-i][x+j] == 255:
                return x+j, y-i
            if edges[y+i][x-j] == 255:
                return x-j, y+i
            if edges[y+i][x+j] == 255:
                return x+j, y+i
    return x, y

# def get_template(frame, pos, railX, railY):
#     template = frame[(railY[pos] - h):railY[pos], railX[pos]:(railX[pos] + w)]
#     return template

def get_template(frame, pos, railX, railY):
    template = frame[(railY[pos] - h):railY[pos], (railX[pos]):(railX[pos] + w)]
    return template


# returns upper stripe of image to be processed with respect to the current template
def get_nextrow(frame, pos, railX, railY):
    row = frame[(railY[pos] - 2 * h):(railY[pos] - h), 0:np.size(frame, 1)]
    return row

def find_next(img, pos, railX, railY, MAX, kk, method=0, weights_on=1):
    w = int(w0 * scale ** pos)
    # methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
    tmpl = get_template(img, pos, railX, railY)
    row = get_nextrow(img, pos, railX, railY)

    # definition of the upper template to be analysed
    startx = railX[pos] - 2 * w

    if startx <= 0:
        startx = 0

    endx = startx + 4 * w

    if endx >= img.shape[1]:
        endx = img.shape[1]

    # Correlation is weighted by a Lorentzian function centred at the peak of correlation given by the lower left corner
    # of the previously extracted template
    xcorr = cv2.matchTemplate(row[0:h, startx:endx], tmpl, method=cv2.TM_CCOEFF_NORMED)

    a = 0.001 * (pos * 2 + 1)  # set Lorentzian shape
    xcorrW = np.zeros_like(xcorr)
    L = []
    val = []
    val.extend(range(0, np.size(xcorr[0])))
    res1 = []
    res2 = []
    for i in range(0, np.size(xcorr, 1)):
        L.append(1 / (1 + a * pow(val[i] - MAX, 2)))
        xcorrW[0][i] = L[i] * xcorr[0][i]
        res1.append([i, xcorr[0][i]])
        res2.append([i, xcorrW[0][i]])

    if pos == 0:
        res = np.array(res1)
    else:
        res = np.array(res2)
    res = res[np.lexsort(-res.T)]

    if len(kk)>0:
        for i in range(len(res)):
            max_loc = res[i][0]
            k = -h / (startx + max_loc - railX[pos] + 10.00001)
            # test.append(abs(k - kk[pos]))
            if abs(k - kk[pos]) < 0.5:
                break
    else:
        max_loc = res[0][0]
    return startx + int(max_loc), railY[pos] - h, [int(max_loc),0]



cap = cv2.VideoCapture("output.avi")#??????????????????

fourcc = cv2.VideoWriter_fourcc(*'XVID')#????????????????????????
out = cv2.VideoWriter('222.avi',fourcc, 30.0, (1920,1080))#??????????????????????????????


# lucas kanade???????????????
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ??????????????????
color = np.random.randint(0,255,(100,3))

# ???????????????
ret, old_frame = cap.read()
#?????????????????????
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
_, Thr_img = cv2.threshold(old_gray, 190, 255, cv2.THRESH_BINARY)  # ????????????????????????210????????????????????????????????????
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # ????????????????????????
edges = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # ??????

#????????????????????????????????????p0???
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

pl = np.array([[[928, 631]]]).astype('float32')
pr = np.array([[[1000, 608]]]).astype('float32')
# p0 = np.array([[[880,641]],[[1060,655]]]).astype('float32')
pl = np.array([[[868, 667]]]).astype('float32')
pr = np.array([[[1061, 697]]]).astype('float32')


# ?????????????????????????????????.
mask = np.zeros_like(old_frame)
count = 1
kl = [-1.111, -1.034, -0.909, -0.882, -0.789, -0.6, -0.4]
kr = [-2.999, -2.999, -2.727, -1.874, -1.578, -1.2, -1.0]
while(1):
    ret,frame = cap.read() #???????????????
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #?????????
    _, Thr_img = cv2.threshold(frame_gray, 190, 255, cv2.THRESH_BINARY)  # ????????????????????????210????????????????????????????????????
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # ????????????????????????
    edges = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # ??????

    # ????????????
    p1, st1, err1 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pl, None, **lk_params)
    p2, st2, err2 = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pr, None, **lk_params)
    # ?????????????????????
    if st1[0]==1:
        good_new = p1[0]
    else:
        good_new = p1[1]
    if st2[0]==1:
        good_new2 = p2[0]
    else:
        good_new2 = p2[1]
    # good_new = p1[st1==1]
    # good_new2 = p2[st2==1]
    lrailx = []
    lraily = []
    rrailx = []
    rraily = []
    pl = []
    pr = []

    new = good_new[0]
    a, b = new.ravel()  # ?????????????????????,???????????????????????????a???b
    lrailx.append(int(a))
    lraily.append(int(b))
    pl.append([[a, b]])
    frame = cv2.circle(frame, (int(a), int(b)), 5, [0,0,255],-1)#??????

    new = good_new2[0]
    a, b = new.ravel()  # ?????????????????????,???????????????????????????a???b
    rrailx.append(int(a))
    rraily.append(int(b))
    pr.append([[a, b]])
    frame = cv2.circle(frame, (int(a), int(b)), 5, [0, 0, 255], -1)  # ??????
    loop = 0
    Ml = (2 * w, frame_gray.shape[0])
    Mr = (2 * w, frame_gray.shape[0])
    point_num = 6
    while (loop<point_num):
        l1, l2, Ml = find_next(frame_gray, loop, lrailx, lraily, Ml[0], kl)
        l1, l2 = jiuzheng(edges, l1, l2)
        lrailx.append(l1)
        lraily.append(l2)
        l3, l4, Mr = find_next(frame_gray, loop, rrailx, rraily, Mr[0], kr)
        l3, l4 = jiuzheng(edges, l3, l4)
        rrailx.append(l3)
        rraily.append(l4)
        if loop == 0:
            pl.append([[l1, l2]])
            pr.append([[l3, l4]])

        cv2.circle(frame, (l1, l2), 5, [0, 0, 255], -1)  # ??????
        cv2.circle(frame, (l3, l4), 5, [0, 0, 255], -1)  # ??????
        # cv2.rectangle(frame, (l1, l2), (l1 + w, l2 - h), (255, 255, 255), 1)
        # cv2.rectangle(frame, (l3, l4), (l3 + w, l4 - h), (255, 255, 255), 1)
        loop = loop + 1
    kl = []
    kr = []

    for i in range(1, len(lrailx)):
        cv2.line(frame, (lrailx[i-1], lraily[i-1]), (lrailx[i], lraily[i]), [0, 0, 255], 5)
        kl.append((lraily[i] - lraily[i-1]) / (lrailx[i] - lrailx[i-1] + 10.00001))
        cv2.line(frame, (rrailx[i - 1], rraily[i - 1]), (rrailx[i], rraily[i]), [0, 0, 255], 5)
        kr.append((rraily[i] - rraily[i - 1]) / (rrailx[i] - rrailx[i - 1] + 10.00001))



    img = cv2.add(frame, mask)  # ????????????????????????????????????
    cv2.imshow('frame',img)  #????????????
    cv2.imwrite('D:/Project/Rail-detection-master/res3/'+str(count)+'.jpg',img)
    count = count + 1

    out.write(img)#?????????????????????

    k = cv2.waitKey(30) & 0xff #???Esc????????????
    if k == 27:
        break

    # ????????????????????????????????????
    pl = np.array(pl).astype('float32')
    pr = np.array(pr).astype('float32')
    old_gray = frame_gray.copy()


out.release()#????????????
cap.release()
cv2.destoryAllWindows()#??????????????????
