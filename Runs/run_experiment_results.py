from TestNetwork.commands import CommandGetResultExperiment, CommandGetAllTest
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import matplotlib.backends.backend_pdf as pdf_manager


print("############ TEST LIST ############")

getTestsCommand = CommandGetAllTest.CommandGetAllTest()

getTestsCommand.execute()

test_list = getTestsCommand.getReturnParam()
for test in test_list:

    print("id:", test.id," || name:", test.name, " || iterations:", test.total_iteration," || period:", test.period," || period new center:", test.period_center," || dt:", test.dt)

print("")  
print("############ SELECT TEST ############")
TEST_ID = int(input("Select test id: "))
MIN_RANGE = int(input("Select min range iteration: "))
MAX_RANGE = int(input("Select max range iteration: "))
PERIOD_ITERATION = int(input("Select iteration period (recommended -> center period): ", ))
AVERAGE_ENERGY = int(input("Show energy as avegare? (0 = no / 1 = yes): " ))

test_name = ""
for test in test_list:
    
    if test.id == TEST_ID:
        test_name = test.name
    
command = CommandGetResultExperiment.CommandGetResultExperiment(testId=TEST_ID)

command.execute(periodIteration=PERIOD_ITERATION, minRange=MIN_RANGE, maxRange=MAX_RANGE)

results = command.getReturnParam()

axis_x = []
axis_y = []
table_data_energy = []
table_data_dna = []

if AVERAGE_ENERGY == 1:

    energy_accum = 0
    for result in results:
        energy_accum += result.tangentPlane.energy

        if result.iteration % PERIOD_ITERATION == 0:
            
            average = energy_accum / PERIOD_ITERATION

            axis_x.append(result.iteration)
            axis_y.append(average)
            table_data_energy.append([result.iteration, average])
            table_data_dna.append([result.iteration, result.dna])

            energy_accum = 0
else:

    for result in results:

        if result.iteration % PERIOD_ITERATION == 0:
            
            axis_x.append(result.iteration)

            if result.tangentPlane.energy is None:
                result.tangentPlane.energy = 0
            axis_y.append(result.tangentPlane.energy)
            table_data_energy.append([result.iteration, result.tangentPlane.energy])
            table_data_dna.append([result.iteration, result.dna])



fig, ax = plt.subplots()
ax.plot(axis_x,axis_y, '.-')

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.8f}"))
ax.title.set_text(test_name+' Graph (Epoch - Energy)')

#y_ticks = np.arange(axis_y[-1], axis_y[0], abs(axis_y[len(axis_y)//2-1] - axis_y[len(axis_y)-1]))
#x_ticks = np.arange(axis_x[0], axis_x[-1]+1, PERIOD_ITERATION)

#plt.yticks(y_ticks)
#plt.xticks(x_ticks)
ax.grid()

for i,j in zip(axis_x,axis_y):
    ax.text(i, j, "{:.10f}".format(j), rotation=45, rotation_mode='anchor', fontsize=8)

fig.set_figheight(8)
fig.set_figwidth(12)

FILE_LOCATION = "reports/"+test_name+".pdf" 
pdf = pdf_manager.PdfPages(FILE_LOCATION)

pdf.savefig(fig, orientation='landscape' )

plt.close()


table = plt.table(cellText=table_data_energy,
                      colLabels=("# Epoch", "Energy"),
                      loc='top', cellLoc='center')


plt.axis('off')
plt.grid('off')

fig = table.figure

#prepare for saving:

# draw canvas once
plt.gcf().canvas.draw()
# get bounding box of table
points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
# add 10 pixel spacing
points[0,:] -= 10; points[1,:] += 10
# get new bounding box in inches
nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)

pdf.savefig(table.figure, orientation='landscape', bbox_inches=nbbox,)

plt.close()

table = plt.table(cellText=table_data_dna,
                      colLabels=("# Epoch", "DNA"),
                      loc='top', cellLoc='center')


plt.axis('off')
plt.grid('off')

fig = table.figure

#prepare for saving:

table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(2)))

# draw canvas once
plt.gcf().canvas.draw()
# get bounding box of table
points = table.get_window_extent(plt.gcf()._cachedRenderer).get_points()
# add 10 pixel spacing
points[0,:] -= 10; points[1,:] += 10
# get new bounding box in inches
nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)

pdf.savefig(table.figure, orientation='landscape', bbox_inches=nbbox,)

plt.close()

pdf.close()

print("PDF Saved succesfully -> Location: ", FILE_LOCATION)
