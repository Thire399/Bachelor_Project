import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D


UNet_baselineVal = [0.8280853218052792, 0.8616764803818968, 0.7881481048475437, 0.8774189028681184,
                     0.7484981445395754, 0.7788797735652151, 0.7954308516011963, 0.750257264641587, 0.7686544766841935]

UNet_baselineTest =[0.7826995201788285, 0.869254936340943, 0.8189312242973832, 0.7883711379503208,
                    0.7591111556347795, 0.7611523645794687, 0.7755866552602796, 0.8176744931109518, 0.8579465052160603,
                    0.8135673414960216, 0.8100743658218111, 0.6675371896526461, 0.8178436946252685, 0.7633568244456402]

ScaleUNetVal = [0.5861429415381482, 0.5332656762654621, 0.6526319220139486, 0.6134836387728118,
                0.5681229137979718, 0.5173388639973859, 0.5884713059712344, 0.6347431290534282, 0.5039389250187614]

ScaleUNetTest = [0.5161679104097556, 0.609077253813316, 0.5624049908559612, 0.49907461477846327, 0.5335370422510655,
                0.5418808766497348, 0.5376293334813776, 0.5108925126699576, 0.5702565760923954,
                0.5345979745385953, 0.4797825217033582, 0.3045407698282008, 0.5560946690234919, 0.5136189991952237]

ScaleNetVal = [0.4867664293942363, 0.4490678711497369, 0.6168272121950925, 0.5849836764805859, 0.5923206689767394,
                0.44229240131639813, 0.5636764929038074, 0.622517092670488, 0.4283326314632996]

ScaleNetTest = [0.454244953075152, 0.5517111937226823, 0.47669208153803744, 0.39287470069475283, 0.4896679681817798,
             0.4885386772407876, 0.4769292611526461, 0.4049959001421819, 0.4645903798103178, 0.42769853192871654,
             0.3670076930638646, 0.23899919365644684, 0.46250885458666446, 0.4608601640557529]

UNet_baselineVal_mean = np.mean(UNet_baselineVal)

UNet_baselineTest_mean = np.mean(UNet_baselineTest)

ScaleUNetVal_mean = np.mean(ScaleUNetVal)

ScaleUNetTest_mean = np.mean(ScaleUNetTest)

ScaleNetVal_mean = np.mean(ScaleNetVal)

ScaleNetTest_mean = np.mean(ScaleNetTest)


labels = ['Val', 'Test', 'Val','Test','Val', 'Test']
#legend = ['UNet Baseline', 'ScaleNet', 'ScaleUNet', 'Image Accuracy']
x_pos = [0, 2, 4, 6, 8, 10]
custom_lines = [Line2D([0], [0], color= 'blue', lw=4),
                Line2D([0], [0], color='green', lw=4),
                Line2D([0], [0], color='purple', lw=4),
                Line2D([0], [0], color='black', lw=4, dash_capstyle = 'round')]
fig, ax = plt.subplots()
ax.scatter([0]*len(UNet_baselineVal), UNet_baselineVal , color = 'black', label = 'Image Accuracy')
ax.scatter([2]*len(UNet_baselineTest), UNet_baselineTest, color = 'black' )
ax.scatter([4]*len(ScaleNetVal), ScaleNetVal, color = 'black' )
ax.scatter([6]*len(ScaleNetTest), ScaleNetTest, color = 'black' )
ax.scatter([8]*len(ScaleUNetVal), ScaleUNetVal, color = 'black' )
ax.scatter([10]*len(ScaleUNetTest), ScaleUNetTest, color = 'black' )

ax.bar(0, UNet_baselineVal_mean, 1,
                yerr= np.std(UNet_baselineVal),
                align='center',
                alpha=0.5,
                color='blue',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))
ax.bar(2, UNet_baselineTest_mean, 1,
                yerr=  np.std(UNet_baselineTest),
                align='center',
                alpha=0.5,
                label = 'UNet Baseline',
                color='blue',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))
ax.bar(4, ScaleNetTest_mean, 1,
                yerr= np.std(ScaleNetVal),
                align='center',
                alpha=0.5,
                color='purple',
                label = 'ScaleNet',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))

ax.bar(6, ScaleNetTest_mean, 1,
                yerr= np.std(ScaleNetTest),
                align='center',
                alpha=0.5,
                color='purple',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))

ax.bar(8, ScaleUNetVal_mean, 1,
                yerr=  np.std(ScaleUNetVal),
                align='center',
                alpha=0.5,
                color='green',
                label = 'ScaleUNet',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))
ax.bar(10, ScaleUNetTest_mean, 1,
                yerr= np.std(ScaleUNetTest),
                align='center',
                alpha=0.5,
                color='green',
                capsize=10,
                error_kw=dict(elinewidth=1, ecolor='red'))



legend = ax.legend(loc = 'upper right', frameon = 1)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Accuracy')
ax.yaxis.grid(True)
ax.set_title('Loss for each image')
plt.tight_layout()
plt.savefig(f'/home/thire399/Documents/Bachelor_Project/Barplt')
plt.show()
