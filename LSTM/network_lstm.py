import torch as torch
import torch.nn as nn
import torch.tensor as tensor

import torch.optim as optim
import LSTM.memory_unit as memory_unit
import Factory.TensorFactory as TensorFactory


class NetworkLSTM(nn.Module):

    def __init__(self, observation_size, in_channels, out_channels, kernel_size, cuda_flag=True):
        super(NetworkLSTM, self).__init__()

        self.cuda_flag = cuda_flag
        self.len_units = observation_size-1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.units = []
        self.xt_unit = []

        self.__createStructure()

    def set_attribute(self, name, value):
        setattr(self,name,value)

    def __get_attribute(self, name):

        attribute = None
        try:
            attribute = getattr(self, name)
        except AttributeError:
            pass

        return attribute

    def delete_attribute(self, name):
        try:
            delattr(self, name)
        except AttributeError:
            pass
    
    # Se crea la estructura de la red neuronal LSTM.
    def __createStructure(self):
        
        for i in range(self.len_units):

            in_channels = self.in_channels
            in_channels_candidate = in_channels

            if i > 0:
                
                in_channels += self.in_channels * 2 
                in_channels_candidate = in_channels_candidate * 2
            
            # Se crea la unidad de memoria.
            internal = memory_unit.MemoryUnit(kernel_size=self.kernel_size, in_channels=in_channels, 
                                                in_channels_candidate=in_channels_candidate, 
                                                out_channels=self.out_channels, cuda_flag=self.cuda_flag)
            self.units.append(internal)
            
            # Se crean los atributos que contiene cada capa convolucional que conforma la unidad de memoria.
            attr_1 = "internal_"+str(i)+"_convFT"
            attr_2 = "internal_"+str(i)+"_convIT"
            attr_3 = "internal_"+str(i)+"_convCND"
            attr_4 = "internal_"+str(i)+"_convOT"

            self.set_attribute(attr_1, internal.convFt)
            self.set_attribute(attr_2, internal.convIt)
            self.set_attribute(attr_3, internal.convCand)
            self.set_attribute(attr_4, internal.convOt)
    
    def __train(self, data, observations):
        
        # Se activa el registro de gradientes.
        self.set_grad_flag(True)
        # Se ejecuta la función de propagación de la red neuronal LSTM.
        self(data)
        # se genera el valor de pérdida.
        self.__generate_loss(observations)
        # Se ejecuta la función de retropropagación.
        self.__backward()
        # Se desactiva el registro de gradientes.
        self.set_grad_flag(False)

    # Params:
    # data = Tensor que contiene los datos de entrenamiento.
    def __create_xt_unit(self, data):
        
        # Se recorre cada unidad de memoria de la red neuronal LSTM y se generan su entrada respectivamente.
        for unit_index in range(self.len_units):
            self.xt_unit.append(self.__get_unit_input(unit_index, data))

        self.xt_unit.append(self.__get_unit_input(self.len_units, data))
    
    def __create_xt_unit_predict(self, data, length):
        
        for unit_index in range(length):
            self.xt_unit.append(self.__get_unit_input(unit_index, data))

    # Params:
    # data = Tensor que contiene los datos de entrenamiento.
    # observations = Lista que contiene las modificaciones y sus pesos (% de precisión) 
    # de cada dato de entrenamiento respectivamente.
    # dt = Tasa de aprendizaje (learning rate) a utilizar para entrenar la red neuronal.
    # p = Cantidad de iteraciones de entrenamiento.
    def Training(self, data, observations, dt=0.1, p=1):
        
        self.xt_unit = []
        
        # Se instancia el optimizador del entrenamiento.
        self.optimizer = optim.SGD(self.parameters(), lr=dt, momentum=0)

        # Se generan las entradas para cada unidad de memoria de la red neuronal.
        self.__create_xt_unit(data)
        i = 0
        print_every = p // 20

        while i < p:
            self.loss = 0

            # Se reinician las gradientes.
            self.optimizer.zero_grad()

            # Se entrena la red neuronal.
            self.__train(data, observations)

            # Se ejecuta el método step del optimizador.
            self.optimizer.step()

            
            if i % print_every == print_every - 1:
                print("L=", self.loss, "i=", i+1)
            
            i += 1
        

    # Params:
    # unit_index = Indice de la unidad de memoria en la lista.
    # data = Tensor que contiene los datos de entrenamiento
    def __get_unit_input(self, unit_index, data):
        shape = data.shape

        if len(shape) > 3:
            value = TensorFactory.createTensorZeros(tupleShape=(shape[0], shape[2], shape[3]), 
                                                        cuda=self.cuda_flag, requiresGrad=False)
        else:
            value = TensorFactory.createTensorZeros(tupleShape=(shape[0], 1, shape[2]), 
                                                        cuda=self.cuda_flag, requiresGrad=False)

        # Se genera la entrada para la unidad de memoria respectiva.
        i = 0
        for word in data:
            letter = word[unit_index]
            value[i] = letter.clone()
            i += 1
        
        return value
            

    def __generate_loss(self, observations):
        
        # Se obtienen los pesos de cada dato de entrenamiento (% de precisión)
        weights = []
        for observation in observations:
            weights.append([[observation.weight]])
        
        weigth_tensor = torch.tensor(weights, dtype=torch.float32, requires_grad=False).cuda()
        unit_index = 0
        loss = 0

        # Se calcula el valor de pérdida utilizando la función de error cuadrático medio ponderado.
        for module in self.units:
            
            value = module.ht - self.xt_unit[unit_index+1]
            value = torch.reshape(value, (value.shape[0], 1, value.shape[1]*value.shape[2]))
            value = torch.mul(value, value)
            value = torch.bmm(weigth_tensor, value)
            loss += value
            unit_index += 1
        
        self.loss = torch.div(loss, self.len_units+1).sum()


    def forward(self, data):
        last_ht = None
        last_ct = None
        unit_index = 0

        # Se ejecuta cada unidad de memoria que conforma la red neuronal LSTM.
        for module in self.units:

            # Se obtiene la entrada de la unidad de memoria respectiva.
            xt = self.xt_unit[unit_index]

            # Se ejecuta la unidad de memoria respectiva.
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct

            unit_index += 1

    def __backward(self):
        self.loss.backward()

    def set_grad_flag(self, flag):

        for module in self.units:

            module.set_grad_flag(flag)
    
    # Params:
    # x = Tensor que contiene los datos necesarios para ejecutar la predicción.
    def predict(self, x):
    
        last_ht = None
        last_ct = None

        modules = x.shape[1]

        self.xt_unit = []

        # Se genera la entrada para cada unidad de memoria de la red neuronal LSTM.
        self.__create_xt_unit_predict(x, modules)

        # Se ejecutan las unidades de memoria que conforman la red neuronal LSTM.
        for unit_index in range(modules):
            xt = self.xt_unit[unit_index]
            module = self.units[unit_index]
            module.compute(xt=xt, last_ht=last_ht, last_ct=last_ct)
            
            last_ht = module.ht
            last_ct = module.ct
        
        # Se obtiene el tensor que contiene los datos resultantes de la predicción.
        ht = self.units[modules-1].ht
        
        return ht
            
