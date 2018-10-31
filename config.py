# config

sample_rate = 22050
n_window = 2048
n_overlap = 1364     # ensure 240 frames in 10 seconds
max_len = 320       # sequence max length is 10 s, 240 frames.
mel_n=128 
step_time_in_sec = float(n_window - n_overlap) / sample_rate

# Name of classes
lbs = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 
       'Dishes', 'Frying', 'Blender', 
       'Running_water', 'Vacuum_cleaner', 
       'Electric_shaver_toothbrush']
          
idx_to_lb = {index: lb for index, lb in enumerate(lbs)}
lb_to_idx = {lb: index for index, lb in enumerate(lbs)}
num_classes = len(lbs)

wav_dir='./features'
out_dir='./features'
recompute=False