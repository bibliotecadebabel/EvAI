import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib.backends.backend_pdf as pdf_manager
from DAO.database.dao import TestModelDAO, TestDAO
import os 

testDAO = TestDAO.TestDAO()
testModelDAO = TestModelDAO.TestModelDAO()
test_list = testDAO.findAll()


tests_to_graph = int(input("Enter the number of tests to graph: "))
selected_test_list = [] 

for i in range(tests_to_graph):
    
    for test in test_list:
        print("id:", test.id,"|| name:", test.name, "|| dt_range:", test.dt, "-", test.dt_min,"|| batchsize:", test.batch_size, "|| date:", test.start_time)

    print("")  
    print("############ SELECT TEST ############")
    TEST_ID = int(input("Select test id: "))

    selected_test = None

    for test in test_list:

        if TEST_ID == test.id:
            selected_test = test
            break

    if selected_test == None:
        raise RuntimeError("Test id does not exist.")

    selected_test_list.append(selected_test)

    print("Selected test:", selected_test.name, "(id:",selected_test.id,")")

max_alai_time = int(input("Enter max alai time to graph: "))

graph_name = "test_graph_result"


fig, ax = plt.subplots()

for selected_test in selected_test_list:

    test_models = testModelDAO.findByLimitAlai(idTest=selected_test.id, limit_alai_time=max_alai_time)
    print(test_models)

    axis_x = []
    axis_y = []
    max_xy = None
    max_y = 0
    for model in test_models:
        axis_x.append(model.current_alai_time/782)
        axis_y.append(model.model_weight)

        if model.model_weight > max_y:
            max_y = model.model_weight
            

    ax.plot(axis_x,axis_y, '-')

ax.text(axis_x[-1], axis_y[-1], "{:.4f}".format(axis_y[-1]), rotation=45, rotation_mode='anchor', fontsize=8)
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.4f}"))
ax.title.set_text(' Graph (Epoch - Accuracy)')

#y_ticks = np.arange(axis_y[-1], axis_y[0], abs(axis_y[len(axis_y)//2-1] - axis_y[len(axis_y)-1]))
#x_ticks = np.arange(axis_x[0], axis_x[-1]+1, PERIOD_ITERATION)

#plt.yticks(y_ticks)
#plt.xticks(x_ticks)
ax.grid()

#for i,j in zip(axis_x,axis_y):
#    ax.text(i, j, "{:.2f}".format(j), rotation=45, rotation_mode='anchor', fontsize=8)

fig.set_figheight(8)
fig.set_figwidth(12)

FILE_NAME = graph_name+".pdf" 
FILE_LOCATION = os.path.join("reports", FILE_NAME)
pdf = pdf_manager.PdfPages(FILE_LOCATION)

pdf.savefig(fig, orientation='landscape' )

plt.close()
pdf.close()