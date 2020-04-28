from TestNetwork.commands import CommandGetResultExperiment, CommandGetAllTest
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
###### EXPERIMENT RESULT SETTINGS ######

# "axis" X
PERIOD_ITERATION = 200






print("############ TEST LIST ############")

getTestsCommand = CommandGetAllTest.CommandGetAllTest()

getTestsCommand.execute()

print("ID || NAME || TOTAL_ITERATIONS || ITERATION_PERIOD || DT")
for test in getTestsCommand.getReturnParam():

    print(test.id," || ", test.name, " || ", test.total_iteration," || ",test.period," || ", test.dt)

print("")  
print("############ SELECT TEST ############")
TEST_ID = int(input("Select test id: "))
TOTAL_ITERATIONS = int(input("Select #totalIterations (0 -> ALL): "))
PERIOD_ITERATION = int(input("Select #iteration period (minium -> selected test's period): ", ))

command = CommandGetResultExperiment.CommandGetResultExperiment(testId=TEST_ID)

command.execute(periodIteration=PERIOD_ITERATION, totalIteration=TOTAL_ITERATIONS)

results = command.getReturnParam()

axis_x = []
axis_y = []
for result in results:
    axis_x.append(result.iteration)
    axis_y.append(result.tangentPlane.energy*1000)


fig, ax = plt.subplots()
ax.plot(axis_x,axis_y)

ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.8f}"))
ax.autoscale(enable=True, axis="y", tight=False)
#plt.plot(axis_x, axis_y, color='lightblue', linewidth=2)
#plt.axis([axis_x[0], axis_x[-1], axis_y[-1], axis_y[0]])

y_ticks = np.arange(axis_y[-1], axis_y[0], 0.05)
x_ticks = np.arange(0, axis_x[-1]+1, PERIOD_ITERATION)
plt.yticks(y_ticks)
plt.xticks(x_ticks)
plt.show()