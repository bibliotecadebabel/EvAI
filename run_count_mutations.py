import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib.backends.backend_pdf as pdf_manager
from DAO.database.dao import TestModelDAO, TestDAO, TestResultDAO
import os 

testDAO = TestDAO.TestDAO()
testModelDAO = TestModelDAO.TestModelDAO()
testResultDAO = TestResultDAO.TestResultDAO()
test_list = testDAO.findAll()

MUTATION_COUNT = 0
def findAccByAlaiRange(min_alai, max_alai, modelList, last_acc):

    acc = last_acc
    for model in modelList:   
        
        if model.current_alai_time >= min_alai and model.current_alai_time <= max_alai:
            acc = model.model_weight
    if acc == 0:
        acc = last_acc
    return acc

tests_to_graph = int(input("Enter the number of tests to graph: "))
selected_test_list = [] 

def convertPosition(y):

    convert = (y-0.7)*(1/0.3)

    return convert


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

#plt.style.use("grayscale")
fig, ax = plt.subplots()

for selected_test in selected_test_list:
    activate_label = True

    if selected_test.id == 7:
        format_line = ".-" 
        label = "First Order"
        color_graph = "#00b33c"
        color_line = "#00802b"
        label_line = "First Order Mutations"
    else:
        format_line = ".-"
        label = "Second Order"
        color_graph = "#0066cc"
        color_line = "#004080"
        label_line = "Second Order Mutations"

    test_graphs = testResultDAO.findByLimitAlai(idTest=selected_test.id, limit_alai=max_alai_time)
    test_models = testModelDAO.findByLimitAlai(idTest=selected_test.id, limit_alai_time=max_alai_time)
    
    axis_x = []
    axis_y = []
    max_xy = None
    max_acc = -1
    last_dna = test_graphs[0].dna
    min_range = 0
    last_acc = 0
    for graph in test_graphs:

        print(graph.id)
        print(graph.dna)
        axis_x.append(graph.current_alai_time / 782)
        acc = findAccByAlaiRange(min_alai=min_range, max_alai=graph.current_alai_time, modelList=test_models, last_acc=last_acc)
        last_acc = acc
        axis_y.append(acc)
        min_range = graph.current_alai_time

        if acc > max_acc:
            max_acc = acc
            max_xy = [graph.current_alai_time / 782, max_acc]
            
        if graph.dna != last_dna:
            MUTATION_COUNT += 1
            min_y = convertPosition(acc*0.990)
            max_y = convertPosition(acc*1.010)

            if activate_label == True:
                activate_label = False
                plt.axvline(x=graph.current_alai_time/782, ls="dotted", ymin=min_y, ymax=max_y, color=color_line, label=label_line)
            else:
                plt.axvline(x=graph.current_alai_time/782, ls="dotted", ymin=min_y, ymax=max_y, color=color_line)

        last_dna = graph.dna
    
    ax.plot(axis_x,axis_y, format_line, label=label, color=color_graph)
    
    print(max_xy)
    ax.text(max_xy[0], max_xy[1], "{:.4f}".format(max_xy[1]), rotation=45, rotation_mode='anchor', fontsize=8)

    print("test: ", selected_test.id)
    print("x: ", axis_x)
    print("y: ", axis_y)

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
ax.title.set_text('First and Second Order Algorithms')

y_ticks = np.arange(0.7, 1.00, 0.05)
x_ticks = np.arange(2, 18, 2)
#x_ticks = np.arange(axis_x[0], axis_x[-1]+1, 5)


plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.xlabel("epochs")
plt.ylabel("validation accuracy")
ax.legend()
ax.grid(False)

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

print("MUTATION COUNT=", MUTATION_COUNT)


