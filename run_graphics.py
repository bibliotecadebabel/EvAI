import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib.backends.backend_pdf as pdf_manager
import random
test_name = "testing"
fig, ax = plt.subplots()

axis_x = []
for i in range(1, 401):
    axis_x.append(782*i)

axis_y = []
for i in range(1, 401):

    if i < 50:
        acc = random.uniform(0.1, 0.4)
    
    if i > 50 and i < 100:
        acc = random.uniform(0.4, 0.7)
    
    if i > 100 and i < 200:
        acc = random.uniform(0.7, 0.85)

    if i > 200 and i < 350:
        acc = random.uniform(0.80, 0.90)
    
    if i > 350:
        acc = random.uniform(0.88, 0.98)

    axis_y.append(acc)

ax.plot(axis_x,axis_y, '-')

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.8f}"))
ax.title.set_text(test_name+' Graph (Epoch - Accuracy)')

#y_ticks = np.arange(axis_y[-1], axis_y[0], abs(axis_y[len(axis_y)//2-1] - axis_y[len(axis_y)-1]))
#x_ticks = np.arange(axis_x[0], axis_x[-1]+1, PERIOD_ITERATION)

#plt.yticks(y_ticks)
#plt.xticks(x_ticks)
ax.grid()

#for i,j in zip(axis_x,axis_y):
#    ax.text(i, j, "{:.2f}".format(j), rotation=45, rotation_mode='anchor', fontsize=8)

fig.set_figheight(8)
fig.set_figwidth(12)

FILE_LOCATION = test_name+".pdf" 
pdf = pdf_manager.PdfPages(FILE_LOCATION)

pdf.savefig(fig, orientation='landscape' )

plt.close()
pdf.close()