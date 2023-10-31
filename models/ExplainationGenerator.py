from .SAINT import SAINT
import numpy as np
import torch

#TODO: REVISAR
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class ExplainationGenerator:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    #TODO: falta forward
    #TODO: falta generate_LRP_last_layer
    #TODO: falta generate_full_lrp
    #TODO: falta generate_attn_last_layer
    #TODO: falta generate_rollout
    #TODO: falta generate_attn_gradcam

    def generateExplanation_all(self, nuevo_categ_enc, nuevo_cont_enc, device):
        reps = self.model.transformer(nuevo_categ_enc, nuevo_cont_enc)
        y_reps = reps[:,0,:]
        output = self.model.mlpfory(y_reps) 

        index = np.argmax(output.cpu().data.numpy(), axis=1) #coge el índice del máximo de la salida del modelo (en nuestro caso debe ser cada fila) para quedarse con la clase elegida
        one_hot = np.zeros((output.size()[0], output.size()[1]), dtype=np.float32) #crea un vector de 0 de tamaño del número de clases que haya (debe ser uno por fila)
        for i in range(one_hot.shape[0]): one_hot[i, index[i]] = 1  #le pone uno a la casilla correspondiente a la clase predicha 

        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1) #para quedarnos solo con el resultado obtenido

        self.model.zero_grad()
        one_hot.backward(torch.ones_like(one_hot), retain_graph=True)
        kwargs = {"alpha": 1}
        self.model.relprop(torch.tensor(one_hot_vector).to(device), **kwargs)

        #TODO: revisar, solo nos quedamos con la relevancia de la atención a columnas, no a filas.
        cams = []
        blocks = self.model.transformer.layers
        #print("NúmeroBloques:", len(blocks))
        for blk in blocks:
            #print("\tNúmero de elementos dentro del bloque: ", len(blk))
            for _ in blk: 
                component = _.fn.fn
                if component.__class__.__name__ == "Attention": 
                    cam = component.get_attn_cam()
                    grad = component.get_attn_gradients()
                    num_features = nuevo_categ_enc.shape[1] + nuevo_cont_enc.shape[1]
                    if cam.shape[2] == num_features and grad.shape[2] == num_features:
                        cams_attn_data = []
                        for num_example in range(0, cam.shape[0]):
                            cam_example = grad[num_example] * cam[num_example]
                            cam_example = cam_example.clamp(min=0).mean(dim=0)
                            cams_attn_data.append(cam_example.unsqueeze(0))
                        cams.append(cams_attn_data)
        
        rollouts_list = []
        for i in range(0, len(cams[0])):
            new_rollout = compute_rollout_attention([cams[0][i]], start_layer=0) #TODO: está hecho solo nos para la relevancia de la atención a columnas, no a filas.
            new_rollout[:, 0, 0] = new_rollout[:, 0].min()
            new_rollout = new_rollout[:, 0]
            #new_rollout = (new_rollout - new_rollout.min()) / (new_rollout.max() - new_rollout.min()) #está quitado porque se hace la normalización después sobre la explicación con minmaxscaler de sklearn
            rollouts_list.append(new_rollout)
            
        rollouts = torch.stack(rollouts_list).view(len(cams[0]), num_features)
        return rollouts, output

    def generateExplanation(self, nuevo_categ_enc, nuevo_cont_enc, device):
        #PARA TRABAJAR SOLO CON EL PRIMER DATO DEL DATALOADER DE TEST, TIENEN QUE TENER UNA PRIMERA DIMENSIÓN IGUAL
        #nuevo_categ_enc = torch.unsqueeze(x_categ_enc[0], dim=0)
        #nuevo_cont_enc = torch.unsqueeze(x_cont_enc[0], dim=0)

        reps = self.model.transformer(nuevo_categ_enc, nuevo_cont_enc)
        y_reps = reps[:,0,:]
        output = self.model.mlpfory(y_reps) 
        
        #PARA TODOS LOS EJEMPLOS DE ENTRADA 
        index = np.argmax(output.cpu().data.numpy(), axis=1) #coge el índice del máximo de la salida del modelo (en nuestro caso debe ser cada fila) para quedarse con la clase elegida
        one_hot = np.zeros((output.size()[0], output.size()[1]), dtype=np.float32) #crea un vector de 0 de tamaño del número de clases que haya (debe ser uno por fila)

        for i in range(one_hot.shape[0]): one_hot[i, index[i]] = 1  #le pone uno a la casilla correspondiente a la clase predicha 

        one_hot_vector = one_hot

        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1) #para quedarnos solo con el resultado obtenido

        self.model.zero_grad()
        one_hot.backward(torch.ones_like(one_hot), retain_graph=True)
        kwargs = {"alpha": 1}
        self.model.relprop(torch.tensor(one_hot_vector).to(device), **kwargs)

        #TODO: revisar, solo nos quedamos con la relevancia de la atención a columnas, no a filas.
        cams = []
        blocks = self.model.transformer.layers
        #print("NúmeroBloques:", len(blocks))
        for blk in blocks:
            #print("\tNúmero de elementos dentro del bloque: ", len(blk))
            for _ in blk: 
                component = _.fn.fn
                if component.__class__.__name__ == "Attention": 
                    cam = component.get_attn_cam()
                    grad = component.get_attn_gradients()
                    num_features = nuevo_categ_enc.shape[1] + nuevo_cont_enc.shape[1]
                    if cam.shape[2] == num_features and grad.shape[2] == num_features:
                        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                        cam = grad * cam
                        cam = cam.clamp(min=0).mean(dim=0)
                        cams.append(cam.unsqueeze(0))

        rollout = compute_rollout_attention(cams, start_layer=0)
        rollout[:, 0, 0] = rollout[:, 0].min()
        return rollout[:, 0], output

        #PARA UN SOLO EJEMPLO DE ENTRADA 
        """output = output[0, :]
        index = np.argmax(output.cpu().data.numpy(), axis=0)
        one_hot = np.zeros((1, output.size()[0]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)"""


