#!/usr/bin/env python

# ale_python_test_pygame_player.py
# Author: Ben Goodrich
#
# This modified ale_python_test_pygame.py to provide a fully interactive experience allowing the player
# to play. RAM Contents, current action, and reward are also displayed.
# keys are:
# arrow keys -> up/down/left/right
# z -> fire button
import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import time
import fcntl
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(threshold=sys.maxsize) # print arrays completely

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)

def preproc_screen(screen_np_in):
    
    screen_image = Image.fromarray(screen_np_in, 'RGB')
    screen_ycbcr = screen_image.convert('YCbCr')
    screen_y = screen_ycbcr.getchannel(0)
    screen_y_84x84 = screen_y.resize((84, 84), resample=Image.BILINEAR)
    screen_y_84x84_float_rescaled_np = np.array(screen_y_84x84, dtype=np.float32)
    return screen_y_84x84_float_rescaled_np

def preproc_screen_score(screen_np_in, pixelScore):  #resolution of score
    
    screen_image = Image.fromarray(screen_np_in, 'RGB')
    screen_ycbcr = screen_image.convert('YCbCr')
    screen_y = screen_ycbcr.getchannel(0)
    screen_y_84x84 = screen_y.resize((pixelScore, pixelScore), resample=Image.BILINEAR)
    screen_y_84x84_float_rescaled_np = np.array(screen_y_84x84, dtype=np.float32)
    
    return screen_y_84x84_float_rescaled_np


if(len(sys.argv) < 2):
    print("Usage ./pygame_master_Vxx.py <ROM_FILE_NAME>")
    sys.exit()

if(len(sys.argv) < 3):
    print("Usage ./pygame_master_Vxx.py <ROM_FILE_NAME> <SESSION_NO>")
    sys.exit()

save_str = sys.argv[2]
print("Session-No.: " + save_str)

session_exist_flag = False

try:

    test_csv = np.loadtxt(open("/workspace/container_mount/ramdisk/responses_vec_" + save_str + ".csv"))
    print("Session-No. already exists! Exiting...")
    session_exist_flag = True

except:

    print("Session-No. checked: New session starting!")

if (session_exist_flag):

    sys.exit()

game_str = sys.argv[1]

if (game_str.find("breakout") == -1):
  #  pixelScore = 104
    scaleFactor  = 2.5
else:
  #  pixelScore =114 #else- Fall momentan Space-inv
    scaleFactor = 1.5
    
pixelScore = 300
    
if (game_str.find("breakout") == -1):

    breakout_flag = False

else:

    breakout_flag = True

if (breakout_flag):

    print("Detected breakout, slowing responses!")

else:

    print("Breakout not detected!")

ale = ALEInterface()

max_frames_per_episode = ale.getInt(b"max_num_frames_per_episode");

np.random.seed()  # initialize numpy to random seed via system clock
ale_seed = np.random.randint(2^32)
ale.setInt(b"random_seed",ale_seed)

random_seed = ale.getInt(b"random_seed")
print("random_seed: " + str(random_seed))

ale.setFloat(b'repeat_action_probability', 0.0)
action_repeat_prob = ale.getFloat(b'repeat_action_probability')
print('action repeat prob.: ' + str(action_repeat_prob))

ale.loadROM(str.encode(sys.argv[1]))
legal_actions = ale.getMinimalActionSet()
print(legal_actions)

(screen_width,screen_height) = ale.getScreenDims()
print("width/height: " +str(screen_width) + "/" + str(screen_height))

#(display_width,display_height) = (1280,840)
(display_width,display_height) = (800,600)

#init pygame
#pygame.init()
pygame.display.init()
pygame.font.init()
# pygame.mixer.init()  # disable sound
#screen = pygame.display.set_mode((display_width,display_height), pygame.FULLSCREEN)
screen = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("Arcade Learning Environment Player Agent Display")

game_surface = pygame.Surface((84,63)) 
game_surface_score = pygame.Surface((pixelScore,80))  #space_inv = breakout = (104,15)
pygame.mouse.set_visible(False)

pygame.display.flip()

#init clock
clock = pygame.time.Clock()

episode = 0
total_reward = 0.0
total_total_reward = 0.0

n_frames = 25200  # must be divisible by 4
human_play_flag = True

loop_count = 0
loop_count_intro = 0
loop_15hz_count = 0
mod_4_count = 0
a = 0
a_old = 0

response_flag_0 = np.zeros((1, 1), dtype=np.uint8)
screen_flag_1 = np.ones((1, 1), dtype=np.uint8)
screen_temp = np.zeros((210, 160, 3), dtype=np.uint8)


#screen_15hz_RGB = np.zeros((int(n_frames/4), 2*210*160), dtype=np.int32)
screen_15hz_RGB = np.zeros((210, 160, 3, 2, int(n_frames/4)), dtype=np.uint8)
responses_vec = np.zeros(n_frames, dtype=np.uint8)
reward_vec = np.zeros(n_frames, dtype=np.double)
episode_vec = np.zeros(n_frames, dtype=np.int32)
trigger_vec = np.zeros(n_frames, dtype=np.int32)
trigger_vec_intro = np.zeros(18000, dtype=np.int32)  # 5 min

waiting_for_trigger_flag = True
trigger_count = 0


if (human_play_flag):
    
    while(waiting_for_trigger_flag):

        pressed = pygame.key.get_pressed()

        if (pressed[pygame.K_t]):

            trigger_count += 1

        trigger_vec_intro[loop_count_intro] = trigger_count

        screen.fill((0,0,0))

        font = pygame.font.SysFont("Ubuntu Mono",32)
        text = font.render("Session-No.: " + save_str, 1, (255,255,255))
        screen.blit(text,(280,110))
        text = font.render("Trigger-No.: " + str(trigger_count), 1, (255,255,255))
        screen.blit(text,(280,210))

        if (trigger_count == 11):

            waiting_for_trigger_flag = False

        pygame.display.flip()

        exit=False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit=True
                break;

        if(pressed[pygame.K_q]):
            exit = True

        if(exit):
            break

        clock.tick(60.)

        loop_count_intro += 1

if (not human_play_flag):
    time_start = time.time()
    
screen_temp_preproc_old  = np.zeros((84,84)) #nur für ersten Durchlauf; aber da max(0,x)=x für x>0 

while(loop_count < n_frames):
    
#while(episode < 10):

    mod_4_count += 1

    #get the keys

    if (human_play_flag):

        keys = 0
        pressed = pygame.key.get_pressed()
        #keys |= pressed[pygame.K_UP]
        #keys |= pressed[pygame.K_DOWN]  <<1
        #keys |= pressed[pygame.K_LEFT]  <<2
        #keys |= pressed[pygame.K_RIGHT] <<3
        #keys |= pressed[pygame.K_z] <<4
        keys |= pressed[pygame.K_u]
        keys |= pressed[pygame.K_2]  <<1
        keys |= pressed[pygame.K_3]  <<2
        keys |= pressed[pygame.K_4] <<3
        keys |= pressed[pygame.K_1] <<4
        a = key_action_tform_table[keys]

        if (pressed[pygame.K_t]):

            trigger_count  += 1


    if (human_play_flag and breakout_flag):

        if (a_old != 0):

            a = 0

        a_old = a

    responses_vec[loop_count] = a
    reward = ale.act(a);
    lives = ale.lives()
    total_reward += reward
    total_total_reward += reward

    reward_vec[loop_count] = total_reward
    episode_vec[loop_count] = episode
    trigger_vec[loop_count] = trigger_count

    #clear screen
    screen.fill((0,0,0))

    #get atari screen pixels and blit them
    numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.uint8)
    ale.getScreenRGB(screen_temp)
    screen_temp_preproc_current = preproc_screen(screen_temp) 
    screen_temp_preproc  = np.maximum(screen_temp_preproc_current , screen_temp_preproc_old)  #damit Schüsse nicht mehr flackern
    #screen_temp_preproc[screen_temp_preproc>26] += 100
    screen_temp_preproc = screen_temp_preproc*1 #Helligkeit Spiel
    screen_temp_reversed = np.reshape(np.dstack((screen_temp_preproc[:,:], screen_temp_preproc[:,:], screen_temp_preproc[:,:], np.zeros((84, 84), dtype=np.uint8)))[:63,:], 63*84*4) 
    numpy_surface[:] = screen_temp_reversed
  
    screen.blit(pygame.transform.scale(game_surface, (800,450)),(0,0))
        
    #screen_temp_preproc_score = preproc_screen_score(screen_temp,pixelScore) 
    #screen_temp_preproc_score = screen_temp_preproc_score*scaleFactor #Helligkeit Score #space_inf *3; breakout *1.5 sonst nichts mehr erkennbar
    #numpy_surface_score = np.frombuffer(game_surface_score.get_buffer(),dtype=np.uint8)
    #screen_temp_reversed_score = np.reshape(np.dstack((screen_temp_preproc_score[:,:], screen_temp_preproc_score[:,:], screen_temp_preproc_score[:,:], np.zeros((pixelScore,pixelScore), dtype=np.uint8)))[220:,:], 80*pixelScore*4) 
    #numpy_surface_score[:] = screen_temp_reversed_score
    
    #screen.blit(pygame.transform.scale(game_surface_score, (800,150)),(0,450)) #Bild in 800x600; 800x80 ist oberer Teil; kann beliebig angepasst werden + damit Koordinaten des ersten 'blit'
    
    #get RAM
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)

    #get current day; steht an 46ter Stelle im RAM
    CurrentDay = "%01X "%ram[45]

    #passed cars: 200/300 Autos steht an 106ter Stelle im RAM und bereits überholte autos an Stelle xx
    #PassedCars = "%03X "%ram[105]

    PassedCars = (int(CurrentDay)-1)*300+200-total_reward

    # convert text to image and then preprocess it
    NewScoreboardTxt = Image.new('RGB',(800,600), color=(0,0,0))
    draw = ImageDraw.Draw(NewScoreboardTxt)
    #font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",25)
    fontBig = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",40)
    daytext = 'Day: ' + CurrentDay
    scoretext = 'Score: ' + str(int(total_reward))
    carstext = 'Cars left: ' + str(int(PassedCars))

    draw.text((50,450), daytext, fill='white', font=font)
    draw.text((50,500), carstext, fill='white', font=font)
    draw.text((50,550), scoretext, fill='white', font=fontBig)

    NewScoreboardImg = np.array(NewScoreboardTxt)[:,:,:3]
    screen_temp_preproc_score = preproc_screen_score(NewScoreboardImg,pixelScore) 
    numpy_surface_score = np.frombuffer(game_surface_score.get_buffer(),dtype=np.uint8)
    screen_temp_reversed_score = np.reshape(np.dstack((screen_temp_preproc_score[:,:], screen_temp_preproc_score[:,:], screen_temp_preproc_score[:,:], np.zeros((pixelScore,pixelScore), dtype=np.uint8)))[220:,:], 80*pixelScore*4) 
    numpy_surface_score[:] = screen_temp_reversed_score
    

    screen.blit(pygame.transform.scale(game_surface_score, (800,150)),(0,450)) #Bild in 800x600; 800x80 ist oberer Teil; kann beliebig angepasst werden + damit Koordinaten des ersten 'blit'
    

    if(mod_4_count == 3):
        
        screen_15hz_RGB[:,:,:,0,loop_15hz_count] = screen_temp

    if(mod_4_count == 4):

        mod_4_count = 0        

        screen_15hz_RGB[:,:,:,1,loop_15hz_count] = screen_temp

        if (not human_play_flag):

            #last_15hz_screen = np.reshape(screen_15hz_RGB[:,:,:,loop_15hz_count], (1,67200))
            last_15hz_screen = screen_15hz_RGB[:,:,:,:,loop_15hz_count]

            #np.savetxt("/media/ramdisk/last_15hz_screen.csv", last_15hz_screen, delimiter=",", fmt='%01.0u')
            np.save('/workspace/container_mount/ramdisk/last_15hz_screen.npy', last_15hz_screen)
            fd_screen = open('/workspace/container_mount/ramdisk/screenlock', 'r')
            fcntl.flock(fd_screen, fcntl.LOCK_EX)
            np.savetxt("/workspace/container_mount/ramdisk/screen_flag.csv", screen_flag_1, fmt='%01.0u')
            fcntl.flock(fd_screen, fcntl.LOCK_UN)
            fd_screen.close()

            repeat_wait_flag = True

            while repeat_wait_flag:
                fd_response = open('/workspace/container_mount/ramdisk/responselock', 'r')
                fcntl.flock(fd_response, fcntl.LOCK_EX)
                response_flag_csv = np.loadtxt(open("/workspace/container_mount/ramdisk/response_flag.csv"))
                fcntl.flock(fd_response, fcntl.LOCK_UN)
                fd_response.close()
                if (response_flag_csv == 1):
                    time_end = time.time()
                    print(loop_count)
                    print(time_end - time_start)
                    print(" ")
                    time_start = time.time()
                    np.savetxt("/workspace/container_mount/ramdisk/response_flag.csv", response_flag_0, fmt='%01.0u')
                    a = np.loadtxt(open("/workspace/container_mount/ramdisk/action_ind_ALE.csv"))
                    repeat_wait_flag = False
                else:
                    time.sleep(0.01)

        loop_15hz_count += 1

    screen_temp_preproc_old  = screen_temp_preproc_current 

    del numpy_surface
    del numpy_surface_score
    #screen.blit(pygame.transform.scale2x(game_surface),(0,0))


    pygame.display.flip()

    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(human_play_flag):
        if(pressed[pygame.K_q]):
            exit = True
    if(exit):
        break

    #delay to 60fps
    clock.tick(60.)

    if(ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) + " Episode Frame Number: " + str(episode_frame_number))
        print("Episode " + str(episode) + " ended with score: " + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1

    loop_count += 1

result_count = 0

while(result_count < 120):

    screen.fill((0,0,0))

    font = pygame.font.SysFont("Ubuntu Mono",32)
    text = font.render("Block " + save_str + " completed!", 1, (255,255,255))
    screen.blit(text,(260,210))
    text = font.render("Total score: " + str(total_total_reward), 1, (255,255,255))
    height = font.get_height()*1.2
    screen.blit(text,(260,210 + height))

    pygame.display.flip()

    #process pygame event queue
    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(human_play_flag):
        if(pressed[pygame.K_q]):
            exit = True
    if(exit):
        break

    clock.tick(30.)

    result_count += 1

screen.fill((0,0,0))

font = pygame.font.SysFont("Ubuntu Mono",32)
text = font.render("Done, writing files...", 1, (255,255,255))
screen.blit(text,(260,210))

pygame.display.flip()

pygame.event.pump()

clock.tick(60.)

#np.savetxt("/media/ramdisk/screen_15hz_RGB_" + save_str + ".csv", screen_15hz_RGB, delimiter=",", fmt='%01.0u')
np.save("/workspace/container_mount/ramdisk/screen_15hz_RGB_" + save_str, screen_15hz_RGB)
np.savetxt("/workspace/container_mount/ramdisk/responses_vec_" + save_str + ".csv", responses_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/reward_vec_" + save_str + ".csv", reward_vec, delimiter=",", fmt='%01.1f')
np.savetxt("/workspace/container_mount/ramdisk/episode_vec_" + save_str + ".csv", episode_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/trigger_vec_" + save_str + ".csv", trigger_vec, delimiter=",", fmt='%01.0u')
np.savetxt("/workspace/container_mount/ramdisk/trigger_vec_intro_" + save_str + ".csv", trigger_vec_intro, delimiter=",", fmt='%01.0u')


