import cv2
import jax
from jax import jit
import timeit
import numpy as np
import os

class FrameExtractor:
    def __init__(self, directory_path, batch_size, key, target_size=(512,300)):
        self.directory_path = directory_path
        self.video_files = [f for f in os.listdir(directory_path) if f.endswith(('.mp4', '.avi', '.npy'))] # Adjust as needed
        self.batch_size = batch_size
        self.key = key
        self.video_gbl_idxs = np.zeros(len(self.video_files)) #holds global idx value for every video 
        self.total_frames = 0
        i = 0

        for f in self.video_files:
            if f.endswith('.npy'):
                frame_count = int(np.shape(np.load(os.path.join(directory_path, f)))[0])
            else:
                frame_count = int(cv2.VideoCapture(os.path.join(directory_path, f)).get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_frames += frame_count
            self.video_gbl_idxs[i] = self.total_frames
            i += 1
        self.cap = None
        self.vid_arr = None
        self.target_size = target_size
        # self.preload_data()
        self.split_jit = jax.jit(jax.random.split)
        self.randomint_jit = jax.jit(jax.random.randint,static_argnames=['shape'])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap is not None:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        

        self.key, idx_key = self.split_jit(self.key)
        idx_array = self.randomint_jit(idx_key, (self.batch_size,), 0, self.total_frames)
        local_idx = 0
        video_idx = 0
        frames = []
        
        for global_idx in idx_array:
            if(global_idx < self.video_gbl_idxs[0]):
                local_idx = int(global_idx)
                #frame from video 0
            else:
                video_idx = np.searchsorted(self.video_gbl_idxs, int(global_idx))
                local_idx = int(global_idx) - int(self.video_gbl_idxs[video_idx-1])
            # print("frame", local_idx)
            # print("video", video_idx)
            vid_pth = self.video_files[video_idx]
            # frame = self.preloaded_data[vid_pth][local_idx]
            # frames.append(frame)
            #Selecting frame for numpy files
            if vid_pth.endswith('.npy'):
                self.vid_arr = np.load(os.path.join(self.directory_path, vid_pth))
                frame = self.vid_arr[local_idx - 1]
                frames.append(frame)
            #Selecting frame for video files
            else:
                self.cap = cv2.VideoCapture(os.path.join(self.directory_path, vid_pth))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
                ret, frame = self.cap.read()
                self.cap.release()

                if ret:
                    frames.append(frame)
       
        array = jax.numpy.array(frames)
        return array.transpose(0,3,2,1)
    
    def preload_data(self):
        self.preloaded_data = {}
        self.i = 1
        for vid_pth in self.video_files:
            full_path = os.path.join(self.directory_path, vid_pth)
            if vid_pth.endswith('.npy'):
                self.preloaded_data[vid_pth] = np.load(full_path)
            else:
                cap = cv2.VideoCapture(full_path)
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
                print("Loaded {} Video", self.i)
                self.i += 1
                self.preloaded_data[vid_pth] = frames
    

def extract_frames(video_path, num_frames, key, target_size=(512, 300)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(str(total_frames)+ " total frames")
    if num_frames > total_frames or num_frames <= 0:
        raise ValueError("Invalid number of frames specified.")

    random_indices = jax.random.randint(key, (num_frames,), 0, total_frames)

    frames = []
    for idx in random_indices:
        ret, frame = cap.read()
        if ret:
            # Resize video to specified target size
            # frame = cv2.resize(frame, target_size)
            frames.append(frame)

    cap.release()

    return jax.numpy.array(frames).transpose(0, 3, 2, 1)




def test_frame_extractor(directory_path, batch_size, key_seed):
    key = jax.random.PRNGKey(key_seed)
    
    
    times = []
    with FrameExtractor(directory_path, batch_size, key) as extractor:
        # Iterate over the frame extractor and display the frames
        overhead_start = timeit.default_timer()
        overhead_end = timeit.default_timer() - overhead_start
        print("Overhead time of  ", overhead_end)

        for i in range(20):
            start_time = timeit.default_timer()
            extractor.__next__().block_until_ready()
            end_time = timeit.default_timer() - start_time
            print(f"Iteration {i} : Time {end_time}\n")
            times.append(round(end_time,3))
    print("Min time of ", min(times))

# def main() -> None:
#     directory_path = "/mnt/disks/persist/vidmod/data/training_resize"
#     batch_size = 120
#     key_seed = 800 
#     test_frame_extractor(directory_path, batch_size, key_seed)  

# if __name__ == '__main__' : 
#     main()