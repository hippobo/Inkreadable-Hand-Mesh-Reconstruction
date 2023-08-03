import cv2
import numpy as np
import mediapipe as mp
import shapely
from shapelysmooth import taubin_smooth
mp_hands = mp.solutions.hands



def get_joints(label_name):
    
   
        
        if label_name == 'Thumb Base':
            chosen_joints = ['Thumb_2' ,'Thumb_3']
        if label_name == 'Thumb Extremity':
            chosen_joints = ['Thumb_3', 'Thumb_4']
        if label_name == 'Index Base':
            chosen_joints = ['Index_2', 'Index_3']
        if label_name == 'Index Extremity':
            chosen_joints = ['Index_3', 'Index_4']

        return chosen_joints

def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h
    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)
    aspect_ratio_scale = new_w/w #because bbox is square
    return scaled_img, aspect_ratio_scale



def get_available_camera():
    for i in range(3, 5):  
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            cap.release()
        else:
            return cap
    return cv2.VideoCapture(0) 



def run_hand_detection(isLeftHand=False):
    size_crop = 224
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    cap = get_available_camera()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    _, frame = cap.read()
    
    joint_indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    h, w, c = frame.shape
    
    s = 10  # number of frames to wait before saving image
    stationary_counter = 0  # counter for stationary frames
    threshold = 10  # threshold to consider as a significant movement
    
    prev_bbox = np.array([0, 0, 0, 0])  # previous bounding box

    while True:
        _, frame = cap.read()

    
    # gray = cv2.medianBlur(gray, 5)
        
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(framergb, cv2.COLOR_RGB2GRAY) 
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 10,
                                param1=100, param2=50,
                                minRadius=5, maxRadius=40)
        
        
        result = hands.process(framergb)
         
      
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks and circles is not None and len(circles)== 1:
            for hand_number, handLMs in enumerate(hand_landmarks):
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for id, lm in enumerate(handLMs.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                    
            #cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                coin_coords = np.array([i[0], i[1], i[2]])
              
                frame_to_save = frame.copy() 
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.circle(frame, center, 1, (0, 100, 100), 3) # circle center
                cv2.circle(frame, center, radius, (255, 0, 255), 3) # circle outline
               
        
                # Check if the bounding box has moved significantly
                curr_bbox = np.array([x_min, y_min, x_max, y_max])
                if np.allclose(prev_bbox, curr_bbox, atol=threshold):
                    stationary_counter += 1
                else:
                    stationary_counter = 0
                prev_bbox = curr_bbox

                # If hand has been stationary for 's' frames, save the image
                if stationary_counter >= s:
                    # Create a slightly bigger bounding box
                    
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    bbox_max_dim = max(bbox_width, bbox_height)
                    bounding_box_scale = 0.5
                    #make bbox square and expand it
                    expand_dim = bounding_box_scale * bbox_max_dim
                    square_bbox = [
                    (x_min + x_max) // 2 - bbox_max_dim // 2 - expand_dim // 2, 
                    (y_min + y_max) // 2 - bbox_max_dim // 2 - expand_dim // 2, 
                    (x_min + x_max) // 2 + bbox_max_dim // 2 + expand_dim // 2, 
                    (y_min + y_max) // 2 + bbox_max_dim // 2 + expand_dim // 2,  
                ]
                    square_bbox = np.clip(square_bbox, 0, [w, h, w, h])

                    x1, y1, x2, y2 = square_bbox.astype(int)
                    hand_image = frame_to_save[y1:y2, x1:x2]

                    
                    w_new = bbox_max_dim + expand_dim
                    h_new = bbox_max_dim + expand_dim

                    
                    print("Picture taken")
                    new_aspect_ratio = size_crop/w_new
                    hand_image_resized = cv2.resize(hand_image,(size_crop,size_crop),interpolation=cv2.INTER_AREA)

                    #hand_image_resized, new_aspect_ratio= resizeAndPad(hand_image, (224, 224), 0)

                    print("aspect ratio : ", new_aspect_ratio)
                    framergb = cv2.cvtColor(cv2.flip(hand_image_resized,1), cv2.COLOR_BGR2RGB)
                    result = hands.process(framergb)
                     
                    if result.multi_handedness:
                        for hand_handedness in result.multi_handedness:
                            label = hand_handedness.classification[0].label
                            print('Handedness:', label)
                            if (label == "Left"):
                                isLeftHand = True
                            else : 
                                isLeftHand = False
                    
                    if isLeftHand:
                        hand_image_resized = cv2.flip(hand_image_resized,1)
                        framergb = cv2.cvtColor(cv2.flip(hand_image_resized,1), cv2.COLOR_BGR2RGB)
                        result = hands.process(framergb)
                 
                    hand_landmarks = result.multi_hand_landmarks
                    world_landmarks = result.multi_hand_world_landmarks
                    joint_coords = []
                    joint_coords_world = []
                    if hand_landmarks:
                        hand_landmark = hand_landmarks[0] #for one hand
                        for joint_num, lm in enumerate(hand_landmark.landmark):
                            if joint_num in joint_indices:
                                x,y = lm.x, lm.y
                                joint_coords.append((x, y))
                    if world_landmarks:
                        world_hand_landmark = world_landmarks[0] #for one hand
                        for joint_num, lm in enumerate(world_hand_landmark.landmark):
                            if joint_num in joint_indices:
                                x,y,z = lm.x, lm.y, lm.z
                                joint_coords_world.append((x, y,z))


                    new_K_matrix = np.array([new_aspect_ratio, x1, y1])

                    # Save the image and reset counter
                    np.save('./samples/hand_info_export/mediapipe_2d_image.npy', joint_coords)
                    np.save('./samples/hand_info_export/coin_coords.npy', coin_coords)
                    np.save('./samples/hand_info_export/new_K_matrix.npy', new_K_matrix)
                    np.save('./samples/hand_info_export/3d_world_coords.npy', joint_coords_world)
                    cv2.imwrite('./samples/hand_cropped/hand_image_cropped.jpg', hand_image_resized)

                    cv2.imwrite('./samples/hand_uncropped/hand_image_uncropped.png', frame_to_save)
                    stationary_counter = 0

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_hand_detection_fp(uncropped_image_file_path,isLeftHand=False):
    def mouse_click(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            # Draw a circle at the clicked point
            if len(points) == 1:
                cv2.circle(frame_copy, points[0], 5, (0, 255, 0), 3)
            elif len(points) == 2:
                radius = int(np.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]))
                cv2.circle(frame_copy, points[0], radius, (0, 255, 0), 3)
    size_crop = 224
    mphands = mp.solutions.hands
    hands = mphands.Hands()
    frame = cv2.imread(uncropped_image_file_path)

   
    joint_indices = list(range(21))
    h, w, c = frame.shape

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(framergb, cv2.COLOR_RGB2GRAY) 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows/10,
                                param1=100, param2=50,
                                minRadius=5, maxRadius=40)
    
    result = hands.process(framergb)
    
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for hand_number, handLMs in enumerate(hand_landmarks):
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lm in enumerate(handLMs.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
        if circles is not None and len(circles) == 1:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # coin center
                radius = i[2]  # coin radius
                coin_coords = np.array([center[0], center[1], radius])
                cv2.circle(frame, center, 1, (0, 100, 100), 3)  # circle center
                cv2.circle(frame, center, radius, (255, 0, 255), 3)  # circle outline
            
        else:
                        
            print("No coin detected. Please click on center and perimeter of coin.")
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', mouse_click)

            points = []  # to store the points clicked by the user
            frame_copy = frame.copy()

            while True:
                cv2.imshow('Image', frame_copy)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or len(points) >= 2:
                    break

            cv2.destroyAllWindows()

            if len(points) < 2:
                print("Not enough points. Please click on the image again.")
            else:
                coin_coords = np.array([points[0][0], points[0][1], np.hypot(points[1][0]-points[0][0], points[1][1]-points[0][1])/2])
                
 
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        bbox_max_dim = max(bbox_width, bbox_height)
        bounding_box_scale = 0.75

        expand_dim = bounding_box_scale * bbox_max_dim
        square_bbox = [
            (x_min + x_max) // 2 - bbox_max_dim // 2 - expand_dim // 2, 
            (y_min + y_max) // 2 - bbox_max_dim // 2 - expand_dim // 2, 
            (x_min + x_max) // 2 + bbox_max_dim // 2 + expand_dim // 2, 
            (y_min + y_max) // 2 + bbox_max_dim // 2 + expand_dim // 2,  
        ]
        square_bbox = np.clip(square_bbox, 0, [w, h, w, h])

        x1, y1, x2, y2 = square_bbox.astype(int)

        color = (0, 255, 0)  # RGB color for the points. Here it is green.
        size = 5  # Size of the points.

        # Draw the points
        # cv2.circle(frame, (x1, y1), size, color, -1)  # Top left corner
        # cv2.circle(frame, (x2, y1), size, color, -1)  # Top right corner
        # cv2.circle(frame, (x1, y2), size, color, -1)  # Bottom left corner
        # cv2.circle(frame, (x2, y2), size, color, -1)  # Bottom right corner

        # # Show the image
        # cv2.imshow('Image with points', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        


        hand_image = frame[y1:y2, x1:x2]

        w_new = bbox_max_dim + expand_dim
        h_new = bbox_max_dim + expand_dim

        new_aspect_ratio = size_crop/w_new
        hand_image_resized = cv2.resize(hand_image,(size_crop,size_crop),interpolation=cv2.INTER_AREA)


        framergb = cv2.cvtColor(cv2.flip(hand_image_resized,1), cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
          
        if result.multi_handedness:
                    for hand_handedness in result.multi_handedness:
                        label = hand_handedness.classification[0].label
                        print('Handedness:', label)
                        if (label == "Left"):
                            isLeftHand = True
                        else : 
                            isLeftHand = False
        
        if isLeftHand:
            hand_image_resized = cv2.flip(hand_image_resized,1)
            framergb = cv2.cvtColor(cv2.flip(hand_image_resized,1), cv2.COLOR_BGR2RGB)
            result = hands.process(framergb)
       

        hand_landmarks = result.multi_hand_landmarks
        world_landmarks = result.multi_hand_world_landmarks
        joint_coords = []
        joint_coords_world = []
        if hand_landmarks:
            hand_landmark = hand_landmarks[0] #for one hand
            for joint_num, lm in enumerate(hand_landmark.landmark):
                if joint_num in joint_indices:
                    x,y = lm.x, lm.y
                    joint_coords.append((x, y))
        if world_landmarks:
            world_hand_landmark = world_landmarks[0] #for one hand
            for joint_num, lm in enumerate(world_hand_landmark.landmark):
                if joint_num in joint_indices:
                    x,y,z = lm.x, lm.y, lm.z
                    joint_coords_world.append((x, y,z))

        
        K_params = np.array([new_aspect_ratio, x1, y1])
        

        # Save the image and reset counter
        np.save('./samples/hand_info_export/mediapipe_2d_image.npy', joint_coords)
        np.save('./samples/hand_info_export/coin_coords.npy', coin_coords)
        np.save('./samples/hand_info_export/new_K_matrix.npy', K_params)
        np.save('./samples/hand_info_export/3d_world_coords.npy', joint_coords_world)

        cv2.imwrite('./samples/hand_cropped/hand_image_cropped.jpg', hand_image_resized)

       

        
    
    else: 
        print("No hand detected. Please choose an image with a visible hand.")
 
  
