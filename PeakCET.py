import os
import cv2
import numpy as np
import pandas as pd
import scipy.io as io
from numpy import linalg as la
from sklearn.cluster import AffinityPropagation
from operator import itemgetter, attrgetter
from scipy.io import loadmat
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import PIL
from scipy.stats import pearsonr
import scipy.io as scio
import tkinter.messagebox as msgbox
from tkinter.ttk import Style
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter
import scipy.signal as sg
import tkinter.font as tkFont
from tkinter import filedialog, dialog,Menu,messagebox
import tkinter as tk
from PIL import ImageFont, ImageDraw, ImageTk, Image
from PIL import Image as imim
from tkinter import *
from tkinter import filedialog
try:
    from PIL import Image
except ImportError:
    import Image


class MForm(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.initComponent(master)

    def initComponent(self, master):
        master.rowconfigure(0, weight=1);
        master.columnconfigure(0, weight=1)
        self.ft = tkFont.Font(family='微软雅黑', size=12, weight='bold')
        self.initMenu(master)
        self.grid(row=0, column=0, sticky=tk.NSEW)
        self.rowconfigure(0, weight=1);
        self.columnconfigure(0, weight=1)

        self.panewin = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        self.panewin.grid(row=0, column=0, sticky=tk.NSEW)

        self.frm_left = ttk.Frame(self.panewin, relief=tk.SUNKEN, padding=0)
        self.frm_left.grid(row=0, column=0, sticky=tk.NS);
        self.panewin.add(self.frm_left, weight=1)
        self.initPlayList()  #

        self.frm_right = ttk.Frame(self.panewin, relief=tk.SUNKEN, style='www.TFrame')
        self.frm_right.grid(row=0, column=0, sticky=tk.NSEW)
        self.frm_right.columnconfigure(0, weight=1);
        self.frm_right.rowconfigure(0, weight=8);
        self.frm_right.rowconfigure(1, weight=2)
        self.panewin.add(self.frm_right, weight=5)
        #第三个
        self.frm_right2 = ttk.Frame(self.panewin, relief=tk.SUNKEN,width=200)
        self.frm_right2.grid(row=0, column=0, sticky=tk.NSEW)
        self.panewin.add(self.frm_right2)
        self.initCtrl()  # 添加滑块及按钮
        #-----------------
        s = ttk.Style();
        s.configure('www.TFrame', background='black')
        img_1 = cv2.imread("1.jpg")
        img_1 = cv2.resize(img_1, (1500, 1100))
        cv2.imwrite(r"...\1.jpg", img_1)
        img_open = imim.open("1.jpg")
        img_png = ImageTk.PhotoImage(img_open)
        self.label_img = tk.Label(self.frm_right, image=img_png)
        self.label_img.image = img_png
        self.label_img.grid()


    def initMenu(self, master):
        '''Initialize Menu'''
        mbar = tk.Menu(master)
        fmenu = tk.Menu(mbar, tearoff=False)
        mbar.add_cascade(label='Input', menu=fmenu, font=('Times', 20, 'bold'))
        fmenu.add_command(label="Open data file", command=self.func1)
        fmenu.add_separator()
        fmenu.add_command(label="Quit", command=root.quit)
        fmenu2 = tk.Menu(mbar, tearoff=False)
        mbar.add_cascade(label='Theme', menu=fmenu2, font=('Times', 20, 'bold'))
        fmenu2.add_command(label="Theme 1", command=self.cod_1)
        fmenu2.add_command(label="Theme 2", command=self.cod_2)
        fmenu2.add_command(label="Theme 3", command=self.cod_3)
        fmenu2.add_separator()
        fmenu2.add_command(label="Quit", command=root.quit)
        master.config(menu=mbar)



    def cod_1(self):
        style = ttk.Style(theme='darkly')
        base = style.master
    def cod_2(self):
        style = ttk.Style(theme='solar')
        base = style.master
    def cod_3(self):
        style = ttk.Style(theme='superhero')
        base = style.master
    def menu_click_event(self):
        pass

    def initPlayList(self):
        self.frm_left.rowconfigure(0, weight=1)  #
        self.frm_left.columnconfigure(0, weight=1)
        tree = ttk.Treeview(self.frm_left, selectmode='browse', show='tree',padding=[10, 10, 0, 0])
        tree.grid(row=0, column=0, sticky=tk.NSEW)  #
        tree.column('#0', width=200)  # 设置图标列的宽度，视图的宽度由所有列的宽决定
        tr_root = tree.insert("", 0, None, open=True, text='List')  # 树视图添加根节点
        node1 = tree.insert(tr_root, 0, None, open=True, text='PeakCET results')
        node11 = tree.insert(node1, 0, None, text='Contours')
        node12 = tree.insert(node1, 1, None, text='Standard-points')
        node13 = tree.insert(node1, 2, None, text='Each contour')
        tree.bind("<<TreeviewSelect>>", self.treeSelect)

    def treeSelect(self, event):
        widgetObj = event.widget
        itemselected = widgetObj.selection()[0]
        col1 = widgetObj.item(itemselected, "text")
        if col1 == 'Contours':
            self.func2()
        elif col1 == 'Standard-points':
            self.func3()
        elif col1 == 'Each contour':
            self.func4()
    def initCtrl(self):
        '''初始化按钮'''
        lab_1 = tk.Button(self.frm_right2, text="Previous", command=self.pre)
        lab_2 = tk.Button(self.frm_right2, text="Next", command=self.submit)
        lab_1.place(x=20, y=550)
        lab_2.place(x=20, y=620)
        lab_3 = tk.Button(self.frm_right2, text="First", command=self.home)
        lab_4 = tk.Button(self.frm_right2, text="Jump", command=self.jumpTo)
        lab_3.place(x=20, y=690)
        lab_4.place(x=20, y=830)
        lab_5 = tk.Button(self.frm_right2, text="AP clustering", command=self.ap_second, font=('calibri', 10, 'bold'))
        lab_5.place(x=20, y=1000)
        self.enTextJumNum = tk.Text(self.frm_right2)
        self.enTextJumNum.place(x=20, y=760, height=45, width=65)
        self.enTextJumNum.insert(tk.INSERT, '0')
    def jumpTo(self):
           global nIndex
           nIndex = int(self.enTextJumNum.get('0.0', 'end'))
           print(nIndex)
           self.submit()
    def submit(self):
        global nIndex
        global photo
        FilePath = r'C:\Users\yy.Xinyue\PycharmProjects\ProjectEdge\png'
        PngList = os.listdir(FilePath)
        strPng = FilePath + '\\' + PngList[nIndex]
        photo = ImageTk.PhotoImage(file=strPng)  # 打开图片看分辨率，如果太大则需要调整
        global imglabel
        nIndex = nIndex + 1
        print("观察单个轮廓:", PngList[nIndex - 1])
        imglabel = tk.Label(self.frm_right, image=photo)
        imglabel.place(x=0, y=0)
        imglabel.config(image=photo)
        l_num = tk.Label(self.frm_right, text=PngList[nIndex - 1]).place(x=1390, y=0)

    # 上一个
    def pre(self):
        global nIndex
        nIndex = nIndex - 2
        print(nIndex)
        self.submit()
   #首页
    def home(self):
        global nIndex
        nIndex = 0
        print(nIndex)
        self.submit()
    def information(self):
        global window3
        window3 = tk.Tk()
        window3.title('Introduction')
        window3.geometry('700x450')
        topLabel = Label(window3, text="This software presents a novel 2D peak detection algorithm called PeakCET, based on Contour Edge Tracking and mass spectra vectors clustering, for fast and effective analysis of two-dimensional (2D) fingerprints."
                                       , font=('宋体', 10)).place(x=20, y=20)
        topLabel.pack()
        window3.mainloop()
    # 数据预处理，归一化
    def transpose_2d(self,data):
        transposed = []
        for i in range(len(data[0])):
            new_row = []
            for row in data:
                new_row.append(row[i])
            transposed.append(new_row)
        r = np.array(transposed)
        return r

    def contours_demo(self,image, ccc, mat_name):
        minval = 5
        maxval = 255
        ret, thresh = cv2.threshold(image, minval, maxval, cv2.THRESH_BINARY)
        cloneImage,contours, heriachy = cv2.findContours(thresh.copy(), cv2.RETR_TREE,
                                                          cv2.CHAIN_APPROX_NONE)  # RETR_TREE包含检测内部
        row = np.array(contours).shape[0]  # 获取行数row
        contours = np.array(contours)
        global new_contours
        new_contours = []
        for j in range(row):
            new_contours.append([])
            for k in range(len(contours[j])):
                if [[contours[j][k][0][0], contours[j][k][0][1]]] not in new_contours[j]:
                    new_contours[j].append([[contours[j][k][0][0], contours[j][k][0][1]]])
        # 删除完成
        hh = []
        out = []
        for j in range(row):
            out.append([])
            for k in range(len(new_contours[j])):
                hh = [new_contours[j][k][0][0], new_contours[j][k][0][1]]
                out[j].append(hh)
        # out存放每个轮廓点的位置
        out = np.array(out)
        feng_number = len(new_contours)
        number = [[0] * (1)] * (feng_number)
        number = np.array(number)
        global area_num
        area_num=[0] * feng_number
        for k in range(feng_number):
            area_num[k]+=len(new_contours[k])
        # io.savemat('area_num.mat', {'array': area_num})
        # G = [0] * row
        # for i in range(1, feng_number):
        #     G[i] = int(G[i - 1] + number[i - 1] + 1)
        # G = np.array(G)

        sc = []
        global intensity
        intensity = []
        for i in range(row):
            intensity.append([])
        for i in range(row):
            for j in range(len(out[i])):
                intensity[i].append(ccc[out[i][j][1]][out[i][j][0]])
        # io.savemat('intensity.mat', {'array': intensity})
        result_xy = []
        # 存放最大值的索引
        idx = []
        for i in range(row):
            value = max(intensity[i])
            idx.append(intensity[i].index(value))
        for i in range(row):
            result_xy.append(out[i][idx[i]])
        # -----------------------------------------------------------------画图
        for j in range(row):
            for k in range(len(contours[j])):
                contours[j][k][0][1] =(y-1) - contours[j][k][0][1]
        # 绘制峰轮廓
        IMAGE = cv2.imread(str(mat_name) + "ori.jpeg")
        IM2 = cv2.resize(IMAGE, (x, y))
        for i, contour in enumerate(contours):
            cv2.drawContours(IM2, contours, i, (0, 0, 255), 1)
        cv2.imwrite(str(mat_name) + "_peaks" + ".png", IM2)
        # 绘制峰标准点
        IM2 = cv2.resize(IMAGE, (x, y))
        for i in range(row):
            cv2.circle(IM2, (result_xy[i][0], ((y-1) - result_xy[i][1])), 2, (0, 0, 255), -1)
        cv2.imwrite(str(mat_name) + "_points" + ".png", IM2)
        global MAX_I
        MAX_I=[]
        for i in range(row):
            MAX_I.append(ccc[result_xy[i][1]][result_xy[i][0]])
        # 输出单个轮廓图片
        for i, contour in enumerate(contours):
            image = cv2.resize(IMAGE, (x, y))
            cv2.drawContours(image, contours, i, (0, 0, 255), 1)
            image = cv2.resize(image, (1500, 1100))
            cv2.imwrite("C:/Users/yy.Xinyue/PycharmProjects/ProjectEdge/png" + "/" + str(i) + ".jpg", image)
        # cv2.imshow("detect contours", IM)
        # new_contours=np.array(new_contours)
        for j in range(row):
            new_contours[j] = np.array(new_contours[j])
        return result_xy, feng_number, out, new_contours

    def trans_normalize(self,R,y_long,y_drift):
        B = R.sum(axis=1)
        #二维漂移， 切片处理
        global y
        y=int(y_long)
        global x
        x=int(len(B)/int(y))
        BB = B.reshape(x, y)
        F = self.transpose_2d(BB)
        io.savemat('F.mat', {'array': F})
        print("--------------------")
        y_drift=int(y_drift)
        y_long=int(y_long)
        # 漂移量为200，进行切片处理
        CA = np.array([[0] * (x)] * (y),dtype=float)
        for q in range(y):
            if q < y_drift:
                for j in range(x):
                    CA[q][j] = F[q + (y-y_drift)][j]  #加250
            else:
                for j in range(x):
                    CA[q][j] = F[q - y_drift][j]  #减150
        # for q in range(y):
        #     for j in range(x):
        #         CA[q][j]=float(CA[q][j])
        minA = CA.min()
        maxA = CA.max()
        N = (CA - minA) / (maxA - minA) * (255 - 0) + 0
        Gray = N.astype(np.uint8)
        return Gray, CA

    def find_index(self,rexy, f):
        ZP_point = np.array([0] * f)
        for i in range(f):
            if rexy[i][1] > y_drift-1:
                ZP_point[i] = y * (rexy[i][0]) + (rexy[i][1]) - y_drift
            else:
                ZP_point[i] = y * (rexy[i][0]) + (rexy[i][1]) + (y-y_drift)
        return ZP_point

    def get_maxima(self,values):
        """极大值点"""
        values = np.array(values)
        max_index = sg.argrelmax(values)[0]
        return max_index, values[max_index]
    # 第二个界面转到
    def ap_second(self):
        image = tk.filedialog.askopenfilenames(title='Choose picture')
        num = len(image)  # 选中的图片数量
        types = ['.jpg', '.png', '.tif', '.gif']
        image_list = list(image)
        global con_num
        con_num = []
        for img in image_list:
            img_info_array = img.split("/")
            local_img_name = img_info_array[-1]
            # 解析图片格式
            name_info = local_img_name.split(".")
            suffix = name_info[-1]
            suffix_len = len(suffix) + 1
            prefix = local_img_name[0:-suffix_len]
            con_num.append(int(prefix))
            print(con_num)
        global window2
        window2 = tk.Tk()
        window2.title('聚类优化结果')
        window2.geometry('1500x800')
        # l2_3 = tk.Label(window2, font=('黑体', 13), width=50, text='')
        # l2_3.pack()
        # l2_3.config(text='已选择%d张待优化的轮廓图片' % num)
        # 逐个进行聚类补齐
        import copy
        m_re=copy.deepcopy(re1)
        in2=copy.deepcopy(MAX_I)
        AREA=copy.deepcopy(area_num)
        for w in range(len(con_num)):
            gu=re1[con_num[w]]
            gu2=MAX_I[con_num[w]]
            gu3=area_num[con_num[w]]
            m_re.remove(gu)
            in2.remove(gu2)
            AREA.remove(gu3)
            print(con_num[w])
        for jk in range(len(con_num)):
            coco = con1[con_num[jk]]
            num = len(coco)
            con_85 = []
            for i in range(len(coco)):
                con_85.append(coco[i][0])
            col = []
            for i in range(num):
                if con_85[i][0] not in col:
                    col.append(con_85[i][0])
            feng_vector = []
            vector_intensity = []
            for i in range(len(col)):
                feng_vector.append([])
                vector_intensity.append([])
            for j1 in range(len(col)):
                for i in range(num):
                    if con_85[i][0] == col[j1]:
                        feng_vector[j1].append(con_85[i])
            for j2 in range(len(col)):
                feng_vector[j2] = sorted(feng_vector[j2], key=itemgetter(1))
                for k in range(len(feng_vector[j2])):
                    vector_intensity[j2].append(ccc1[feng_vector[j2][k][1]][feng_vector[j2][k][0]])
            max_value = []
            max_index = []
            max_xy = []
            for i in range(len(col)):
                max_value.append(self.get_maxima(vector_intensity[i])[1].tolist())
                max_xy.append([])
                max_index.append([])  # 找出轮廓内所有列的极大值
            for i in range(len(col)):
                if max_value[i] != []:
                    for j in range(len(max_value[i])):
                        max_index[i].append(vector_intensity[i].index(max_value[i][j]))
            for i in range(len(col)):
                if max_value[i] != []:
                    for j in range(len(max_value[i])):
                        max_xy[i].append(feng_vector[i][max_index[i][j]])
            end_xy = []
            for i in range(len(col)):
                if max_value[i] != []:
                    for j in range(len(max_value[i])):
                        end_xy.append(max_xy[i][j])
            max_zp = [[0] * 365] * len(end_xy)
            max_zpindex = self.find_index(end_xy, len(end_xy))
            for i in range(len(end_xy)):
                max_zp[i] = (R1[ max_zpindex[i],:])
            result_zp = []
            for i in range(len(end_xy)):
                result_zp.append(max_zp[i])
            # AP向量聚类，自动得到聚类中心
            clustering = AffinityPropagation(random_state=5, damping=0.5).fit(result_zp)
            arr = clustering.labels_
            center = clustering.cluster_centers_

            new_center = []
            for i in range(len(center)):
                new_center.append(center[i, :])
            long = len(center)
            center_index = []
            for i in range(long):
                center_index.append([])
            for i in range(long):
                b = new_center[i]
                for m in range(len(end_xy)):
                    a = max_zp[m]
                    center_index[i].append((a == b).all())
            # 得到聚类中心向量在所有极大值点质谱向量的位置索引
            res_index = []
            for i in range(long):
                res_index.append(center_index[i].index(True))
            add_xy = []
            for i in range(long):
                add_xy.append(end_xy[res_index[i]].tolist())
            center_num = long
            new_xy = add_xy
            print("补齐的峰", new_xy)
            print(jk)
            print("hahahahha",con_num[jk])
            print("总面积",len(new_contours[con_num[jk]]))
            for k in range(center_num):
                m_re.append(new_xy[k])
                in2.append(ccc1[new_xy[k][1]][new_xy[k][0]])
                AREA.append(len(new_contours[con_num[jk]])/center_num)
            print("hahahaha")

        # 可视化补充后的峰
        img = cv2.imread(str(mat_name) + "ori.jpeg")
        imgnew = cv2.resize(img, (x, y))
        for i in range(len(m_re)):
            cv2.circle(imgnew, (m_re[i][0], ((y-1) - m_re[i][1])), 2, (0, 0, 255), -1)
        imgnew = cv2.resize(imgnew, (1500, 800))
        cv2.imwrite(str(mat_name)+'update_p.png', imgnew)
        # 展示
        my_image3 = PhotoImage(file=str(mat_name)+'update_p.png', master=window2)
        imglabel_3 = tk.Label(window2, image=my_image3)
        imglabel_3.place(x=0, y=0)
        # ========================
        print("优化后的峰数量", len(m_re))
        io.savemat('ap_result.mat', {'array': m_re})
        io.savemat('MAX_intensity.mat', {'array': in2})
        io.savemat('area_num.mat', {'array': AREA})
        print("聚类优化完成！")
        window2.mainloop()
    def func1(self):
        global mat_name
        # 获取原始数据
        global file_path
        file_path = filedialog.askopenfilename()
        print('打开文件：', file_path)
        mat_name = file_path
        window1 = tk.Tk()
        window1.title('Visualization condition')
        window1.geometry('500x350')
        tk.Label(window1, width=10).grid()
        tk.Label(window1, text='2D sampling points').grid(row=0, column=0, sticky=tk.W, pady=20)
        tk.Label(window1, text='2D slice drift').grid(row=1, column=0, sticky=tk.W, pady=20)
        global e1
        global e2
        e1=tk.Entry(window1)
        e2 = tk.Entry(window1)
        e1.grid(row=0, column=1)
        e2.grid(row=1, column=1)
        b1 = tk.Button(window1, text="Enter",command=self.func1_2)  # 使用 ttkbootstrap 的组件
        b1.grid(row=2, column=1, sticky=tk.W, pady=20)

    def func1_2(self):
        global data
        global re1
        global con1
        global ccc1
        global R1
        global y_long
        global y_drift
        y_long = int(e1.get())
        y_drift = int(e2.get())
        data = loadmat(mat_name)
        R1 = data['AAA']
        gray1, ccc1 = self.trans_normalize(R1,y_long,y_drift)
        Z = self.transpose_2d(ccc1)
        io.savemat('CCC.mat', {'array': ccc1})
        fig, (ax0) = plt.subplots(1, 1)
        c = ax0.pcolor(Z.T, cmap='jet')
        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去 x 轴刻度
        plt.yticks([])
        fig.tight_layout()
        plt.savefig(str(mat_name) + "ori.jpeg", bbox_inches='tight', pad_inches=0, dpi=500)
        ori = cv2.imread(str(mat_name)+ "ori.jpeg")
        ori_pic = cv2.resize(ori, (x, y))
        cv2.imwrite(str(mat_name) + "ori.jpeg", ori_pic)
        plt.show()
        re1, F1, out_1, con1 = self.contours_demo(gray1, ccc1, mat_name)
        point_1 = self.find_index(re1, F1)
        print("文件成功传入，生成原始图片，初步峰检测完成")
        print("轮廓检测到的峰数量为",len(re1))
        io.savemat('con_result.mat', {'array': re1})

        return re1, ccc1, con1

    def func2(self):
        print("展示峰轮廓图")
        #调整展示大小
        show_im1=cv2.imread(str(mat_name) + "_peaks" + ".png")
        show_im1 = cv2.resize(show_im1, (1500, 1100))
        cv2.imwrite(str(mat_name) + "_peaks" + ".png", show_im1)
        #-----------------
        im1 = imim.open(str(mat_name) + "_peaks" + ".png")
        img_png2 = ImageTk.PhotoImage(im1)
        self.label_img.config(image=img_png2)  # 定义label
        self.label_img.image = img_png2  # 要重新进行赋值
        self.label_img.grid()

    def func3(self):
        print("展示峰标准点")
        show_im2 = cv2.imread(str(mat_name) + "_points" + ".png")
        show_im2 = cv2.resize(show_im2, (1500, 1100))
        cv2.imwrite(str(mat_name) + "_points" + ".png", show_im2)
        im2 = imim.open(str(mat_name) + "_points" + ".png")
        img_png3 = ImageTk.PhotoImage(im2)
        self.label_img.config(image=img_png3)  # 定义label
        self.label_img.image = img_png3  # 要重新进行赋值
        self.label_img.grid()
    def func4(self):
        img = Image.open(r'C:\Users\yy.Xinyue\PycharmProjects\ProjectEdge\png\0.jpg')
        # 打开第一张轮廓图片
        img_one = ImageTk.PhotoImage(img)
        self.label_img.config(image=img_one)  # 定义label
        self.label_img.image = img_one # 要重新进行赋值
        self.label_img.grid()

if (__name__ == '__main__'):
    root = ttk.Window()
    root.geometry('1900x1100')
    root.title('PeakCET')
    root.minsize(800, 480)
    app = MForm(root)
    nIndex = 0
    root.mainloop()
