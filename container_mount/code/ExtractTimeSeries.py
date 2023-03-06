from ale_python_interface import ALEInterface
import numpy as np
from PIL import Image
import torch
from torch import nn
import copy
import time
import fcntl

#noch ist alles im gleichen Ordner; Pfade müssen geändert werden
#manuelle Eingaben
weightNumber = 40
game  = 'space_invaders'
SessionNumber = 1

class DqnNN(nn.Module):
    
    def __init__(self, no_of_actions):
        super(DqnNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, no_of_actions))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out
		
def ale_init(rng): #brauch ich das überhaupt?
    
    max_frames_per_episode = 50000
    
    ale = ALEInterface()
    ale.loadROM(str.encode('/workspace/container_mount/roms/'+game+'.bin'))
    ale.setInt(b'max_num_frames_per_episode', max_frames_per_episode)
    minimal_actions = ale.getMinimalActionSet()
    (screen_width,screen_height) = ale.getScreenDims()
    ale_seed = rng.integers(2^32)
    ale.setInt(b'random_seed', ale_seed)
    random_seed = ale.getInt(b'random_seed')
    ale.setFloat(b'repeat_action_probability', 0.0)
    action_repeat_prob = ale.getFloat(b'repeat_action_probability')
   
    return ale
        
def ale_15hz(last_15hz_screen): #gekürzt
    
    screen_vec_1 = last_15hz_screen[:,:,:,0]
    screen_vec_2 = last_15hz_screen[:,:,:,1]
    
    screen_R_max = np.amax(np.dstack((screen_vec_1[:,:,0], screen_vec_2[:,:,0])), axis=2)
    screen_G_max = np.amax(np.dstack((screen_vec_1[:,:,1], screen_vec_2[:,:,1])), axis=2)
    screen_B_max = np.amax(np.dstack((screen_vec_1[:,:,2], screen_vec_2[:,:,2])), axis=2)
    
    screen_max = np.dstack((screen_R_max, screen_G_max, screen_B_max))
    
    return screen_max

def preproc_screen(screen_np_in):
    
    screen_image = Image.fromarray(screen_np_in, 'RGB')
    screen_ycbcr = screen_image.convert('YCbCr')
    screen_y = screen_ycbcr.getchannel(0)
    screen_y_84x84 = screen_y.resize((84, 84), resample=Image.BILINEAR)
    screen_y_84x84_float_rescaled_np = np.array(screen_y_84x84, dtype=np.float32)/255.0
    
    return screen_y_84x84_float_rescaled_np

class DqnAgent(): #gekürzt
    
    def __init__(self, rng, minimal_actions):
        
        self.network = DqnNN(len(minimal_actions)).to('cuda')
        self.minimal_actions = minimal_actions
        self.state = np.zeros((4, 84, 84), dtype=np.float32)
  
    def __call__(self, screen, reward, train_flag):
        
        state_new = self.update_state(screen)
        self.state = state_new
        
        return state_new
    
    def update_state(self, screen): #?
    
        state_new = np.concatenate((np.expand_dims(screen, axis=0), self.state[0:3, :,:]))
        
        return state_new

#hook: fancy function:  extract features from layers of the NN for each  forward-pass
def get_activation_value(name):
    
    def hook(m, input, output):
        actIn[name] = input[0].detach()
        actOut[name] = output.detach()
        
    return hook
    
#Main

rng = np.random.default_rng()
ale = ale_init(rng)
minimal_actions = ale.getMinimalActionSet()
dqn_agent = DqnAgent(rng, minimal_actions)

#load DNN weights 
dqn_agent.network.load_state_dict(torch.load(game+'_DNN_epoch_'+str(weightNumber)+'.pt'))
dqn_agent.network.eval()

#load video
Video15Hz = np.load('screen_15hz_RGB_'+str(SessionNumber)+'.npy') #(210,160,3,2,6300)
NumberOfFrames15Hz = 6300

#inizialise arrays
ActValuesLayerOne = np.zeros((NumberOfFrames15Hz,1,32,20,20))
ActValuesLayerThree = np.zeros((NumberOfFrames15Hz,1,64,9,9))
ActValuesLayerFive = np.zeros((NumberOfFrames15Hz,1,64,7,7))
ActValuesLayerEight = np.zeros((NumberOfFrames15Hz,1,512))
ActValuesLayerTen = np.zeros((NumberOfFrames15Hz,1,6))

for i in range(NumberOfFrames15Hz):
    
    Screen15HzCurrent = Video15Hz[:,:,:,:,i]
    frame = preproc_screen(ale_15hz(Screen15HzCurrent)) 
    state_new = dqn_agent(frame, 0, False) #hier wird update_state angewandt
    state_tensor = torch.from_numpy(state_new).cuda().unsqueeze(0) #ready for NN
    actIn = {}
    actOut = {}
    
    #'register', that after each forward() a forward hook should be performed again
    for name, module in dqn_agent.network.conv.named_children(): #name ist hier nur '0','1','2','3'; module ist hier Conv2d,Relu(),....
        
        h=module.register_forward_hook(get_activation_value(name)) #nach forward wird hook-Funktion mit input und output ausgeführt, kann garantiert auch außerhalb der Schleife gemacht werden
        
    out  = dqn_agent.network(state_tensor)   #run for trigger     
    h.remove() #hook will no longer trigger
    
    #for j in range(10):
        #print('Shape:', actOut[str(j)].shape)
  
    ##store for each time step in dict numpy-format
    #NumpyActOutDict={}
   # for j in range(10): #10 layers
       # NumpyActOut = act_out[str(j)].numpy()
      #  NumpyActOutDict[str(j)] = NumpyActOut        
    #np.save('file.npy',NumpyActOutDict) #save as ndict with numpy array

    #store for each time step in dict torch-format
    #torch.save(actOut,'file.pt') #save as torch tensor      
        
    #store values for each layer (bevor applying a nonlinear function
    ActValuesLayerOne[i] = actOut['0'].cpu().numpy() #Conv2d; Inputlayer
    ActValuesLayerThree[i] = actOut['2'].cpu().numpy() #Conv2d
    ActValuesLayerFive[i] = actOut['4'].cpu().numpy() #Conv2d
    ActValuesLayerEight[i] = actOut['7'].cpu().numpy() #Linear
    ActValuesLayerTen[i] = actOut['9'].cpu().numpy() #Linear; Outputlayer
   
    np.save('Layer1_Session'+str(SessionNumber)+'.npy',ActValuesLayerOne) 
    np.save('Layer3_Session'+str(SessionNumber)+'.npy',ActValuesLayerThree) 
    np.save('Layer5_Session'+str(SessionNumber)+'.npy',ActValuesLayerFive) 
    np.save('Layer8_Session'+str(SessionNumber)+'.npy',ActValuesLayerEight) 
    np.save('Layer10_Session'+str(SessionNumber)+'.npy',ActValuesLayerTen) 
    
