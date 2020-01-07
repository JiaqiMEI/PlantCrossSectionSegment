# 本文件主要为该项目“全局变量”集合
#   文件头初始化的全局变量，在具体方法中赋值，随后由另一文件调用，出错，暂不明缘由
#   故建立独立文件初始化所有“全局变量”，之后都从该独立文件调用
# 以及复用率高的方法集合

import numpy as np
import wx

# 可见光图像
Visimage = np.array([])
grayimage = np.array([])

#保留原始数据，以便重选前景提取方案
VisimageO = np.array([])
grayimageO = np.array([])

# 亮度均衡化，灰度直方图均衡化
dstGray = np.array([])
VisimageYUV = np.array([])
visNew = np.array([])

#分割用数据
visGray = np.array([])
#表皮厚度
thicky = np.array([])
# 组织分离结果
EE = np.array([])
VV = np.array([])
SS = np.array([])
PP = np.array([])
AreaEE = np.array([])
AreaVV = np.array([])
AreaSS = np.array([])
AreaPP = np.array([])

# fore，前景传递参数
fore = np.array([])
foreP = np.array([])
fore2 = np.array([])

# 前景提取算法结果
iteration = np.array([])
homogeneity = np.array([])
distance = np.array([])
local = np.array([])
Otsu = np.array([])
outer = np.array([])

# 以threshold为阈值进行分割，计算分割类别的均值,数据量占比
# 目前针对numpy数组
def arr_mean2(arr_a, threshold):
    thp = arr_a >= threshold
    th = np.multiply(arr_a, thp)

    if np.sum(thp)==0:
        mean_h = 0
    else:
        mean_h = np.sum(th)/np.sum(thp)

    thp_r = np.sum(thp)/arr_a.size

    tlp = arr_a < threshold
    tl = np.multiply(arr_a, tlp)
    if np.sum(tlp)==0:
        mean_l = 0
    else:
        mean_l = np.sum(tl)/np.sum(tlp)
    tlp_r = np.sum(tlp)/arr_a.size

    return mean_h, mean_l, thp_r, tlp_r

# 归一化至指定区间
# mapminmax
def mapminmax_(a, min_, max_):
    b = np.min(a)
    c = np.max(a) - np.min(a)
    d = (max_ - min_)*(a - b) / c
    return d

# 前景提取算法6种
# Iteration算法
def Iteration(a, k1, k2, tol):
    # 阈值初始化
    T = k1* (a.min() + a.max())
    done = False

    # 进度条怎么加呢？不知道总次数。。。

    while ~done:
        meanTh, meanTl, p2, p1 = arr_mean2(a, T)

        # 计算新阈值
        Tn = k2 * (meanTh + meanTl)
        done = abs(T - Tn) < tol
        T = Tn
    del meanTl, meanTh, p2, p1
    del done

    # 以阈值T，二值化
    iteration = np.copy(a)
    iteration[iteration < T] = 0
    iteration[iteration >= T] = 1
    return iteration

# homogeneity
def Homogeneity(a, dt):
    Smin = -1
    T = 0
    p = a.min()
    q = a.max()
    m,n = a.shape

    # 进度条
    progressMax = dt+2
    progressdlg = wx.ProgressDialog("The Progress(Homogeneity)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1
    for TT in np.arange(p, q, (q - p) / 1000):
        ave2, ave1, p2, p1 = arr_mean2(a, TT)

        # 计算d1,d2,为两部分与各自均值的差的平方的和
        d1, d2 = -1, -1
        for ii in np.arange(0, m, 1):
            for jj in np.arange(0, n, 1):
                if a[ii, jj] >= TT:
                    d = (a[ii, jj] - ave2) ** 2
                    if d2 == -1:
                        d2 = d
                    else:
                        d2 = d2 + d
                else:
                    d = (a[ii, jj] - ave1) ** 2
                    if d1 == -1:
                        d1 = d
                    else:
                        d1 = d1 + d

        del ave1, ave2

        S = p1 * d1 + p2 * d2

        del p1, p2

        if Smin == -1:
            Smin = S
            T = TT
        else:
            if S < Smin:
                Smin = S
                T = TT

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    #print(T)
    # 以阈值T，二值化
    homogeneity = np.copy(a)
    homogeneity[a >= T] = 1
    homogeneity[a < T] = 0

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return homogeneity

# The maximum distance between classes
def Distance(a, dt):
    Smax = 0
    T = 0
    p = a.min()
    q = a.max()

    # 进度条
    progressMax = dt + 2
    progressdlg = wx.ProgressDialog("The Progress(max_distance)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for TT in np.arange(p, q, (p + q) / dt):
        aveH, aveL, p2, p1 = arr_mean2(a, TT)
        S = ((aveH - TT) * (TT - aveL)) / (aveH - aveL) ** 2
        if S > Smax:
            Smax = S
            T = TT

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    # 以阈值T，二值化
    distance = np.copy(a)
    distance[distance < T] = 0
    distance[distance >= T] = 1

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return distance

# Local Threshold
def LocalThreshold(a,ee,kk):
    # 对每个子区域，求阈值并二值化
    b = np.copy(a)
    m, n = a.shape

    # 进度条
    progressMax = ee*kk + 2
    progressdlg = wx.ProgressDialog("The Progress(Local Thresholding)", "Time remaining", progressMax,
                                style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for ii in range(ee):
        for jj in range(kk):
            d = b[int(ii * m / ee):int((ii + 1) * m / ee), int(jj * n / kk):int((jj + 1) * n / kk)]
            # 以子区域均值为子区域阈值
            d[d < d.mean()] = 0
            d[d >= d.mean()] = 1

            count = count + 1
            if keepGoing and count < progressMax:
                keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return b

# Otsu
def otsu_(a):
    a = mapminmax_(a, 0, 255)
    max_g = 0
    T = 0

    # 进度条
    progressMax = 256 + 2
    progressdlg = wx.ProgressDialog("The Progress(Otsu)", "Time remaining", progressMax,
                                    style=wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME | wx.PD_REMAINING_TIME)
    keepGoing = True
    count = 1

    for Th in range(0, 256):
        foreAve, bgAve, foreRatio, bgRatio = arr_mean2(a, Th)
        g = foreRatio*bgRatio*(foreAve-bgAve)*(foreAve-bgAve)
        if g > max_g:
            max_g = g
            T = Th

        count = count + 1
        if keepGoing and count < progressMax:
            keepGoing = progressdlg.Update(count)

    # 以T为阈值，二值化
    Otsu = a
    Otsu[Otsu < T] = 0
    Otsu[Otsu >= T] = 1

    count = count + 1
    if keepGoing and count < progressMax:
        keepGoing = progressdlg.Update(count)

    progressdlg.Destroy()
    return Otsu

# Cluster
#Epidermis = np.zeros((m, n))
#VascularBundle = np.zeros((m, n))
#Sclerenchyma = np.zeros((m, n))
#Parenchyma = np.zeros((m, n))

# wxPython可能有框架限制，对象A实例化对象B时，无法通过参数传递的方式将参数应用于对象B的初始化方法
# 目前发现两种可行的方法：
# 1.在类B的初始化方法之外建立一个“全局变量”，如a=None；
#    从对象A实例化对象B，如A.k=B();
#    从对象A为对象B的“全局变量”赋值，如A.k.a=1;
#       针对上述情况，若有对象C再次实例化对象B，B.a=1这一结论仍然存在，除非有对B.a另行赋值操作
# 2.在类B的初始化方法之外新建方法，如def setParent(self, parent)：self.parent = parent
#    从对象A实例化对象B，如A.k=B()
#    从对象A调用对象B的新建方法，如B.setParent(p), 则B.parent = p
#       针对上述情况，B.parent = p是该对象B的新增属性；若有对象C再次实例化对象B，B.parent这一属性不存在

# 矩阵左除，右除
# 左除：a/b = a*inv(b)
# 右除：a\b = inv(a)*b
# speye(m, n) % 生成m×n的单位稀疏矩阵；speye(n) % 生成n×n的单位稀疏矩阵——————matlab
# The functions spkron, speye, spidentity, lil_eye and lil_diags were removed from scipy.sparse.
# The first three functions are still available as scipy.sparse.kron, scipy.sparse.eye and scipy.sparse.identity.
#starttime = time.time()      #Python time time() 返回当前时间的时间戳（1970纪元后经过的浮点秒数）
#maxtime = 600                # 控制计算用时至多为600s
#Python3.8 没有time.clock()了

# 不要随意妄想暂停进程/线程
# 需要子窗口返回值(主窗口暂停执行)就把子窗口设定为Dialog
