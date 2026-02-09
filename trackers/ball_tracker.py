import pickle
from ultralytics import YOLO
import cv2
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_ball_shot_frames(self,ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions['ball_hit'] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25

        for i in range(1, len(df_ball_positions) - minimum_change_frames_for_hit):

            curr = df_ball_positions['delta_y'].iloc[i]
            next_ = df_ball_positions['delta_y'].iloc[i + 1]

            direction_change = (curr > 0 and next_ < 0) or (curr < 0 and next_ > 0)

            if direction_change:
                change_count = 0

                for j in range(i + 1, i + minimum_change_frames_for_hit):
                    prev_dy = df_ball_positions['delta_y'].iloc[j - 1]
                    curr_dy = df_ball_positions['delta_y'].iloc[j]

                    if curr > 0 and curr_dy < 0:
                        change_count += 1
                    elif curr < 0 and curr_dy > 0:
                        change_count += 1

                if change_count >= minimum_change_frames_for_hit * 0.6:
                    df_ball_positions.loc[df_ball_positions.index[i], 'ball_hit'] = 1
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def interpolate_ball_position(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]

        # create dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # interpolate the missing valuees
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections =[] 

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            dict_player = self.detect_player(frame)
            ball_detections.append(dict_player)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)
        return ball_detections



    def detect_player(self, frames):
        results = self.model.predict(frames,conf = 0.5)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict
    
    def draw_bbox(self, video_frames, detections):
        output_frames = []
        for frame, ball_dict in zip (video_frames, detections):
            
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            output_frames.append(frame)
        return output_frames
    
    

       