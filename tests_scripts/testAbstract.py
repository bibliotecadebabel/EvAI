from DAO import GeneratorFromCIFAR, GeneratorFromImage, GeneratorFromImageNet


dataGenImage = GeneratorFromImage.GeneratorFromImage(2, 120)
#dataGenImageNet = GeneratorFromImageNet.GeneratorFromImageNet(5, 200)
#dataGenCIFAR = GeneratorFromCIFAR.GeneratorFromCIFAR(5, 250)

dataGenImage.dataConv3d()
#dataGenImage.dataConv3d()
#dataGenImage.dataNumpy()

print(dataGenImage.data[0][0].shape)

dataGenImage.dataNumpy()
print(dataGenImage.data[0][0].shape)