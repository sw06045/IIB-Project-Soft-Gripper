#! /usr/bin/env python3

import math
import roslib
import rospy
import foodly_move_msgs
import actionlib
import geometry_msgs
import cv2
import numpy as np
import os, sys
import torch
import time
import serial
import json
import re


import tensorflow as tf

from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped




from cv_bridge import CvBridge, CvBridgeError
from foodly_move_msgs.msg import Camera
from foodly_move_msgs.msg import Indicator
from foodly_move_msgs.msg import CancelPlan
from foodly_move_msgs.msg import Pose
from foodly_move_msgs.msg import EndeffectorAction
from foodly_move_msgs.msg import EndeffectorGoal
from foodly_move_msgs.msg import EndeffectorFeedback
from foodly_move_msgs.msg import EndeffectorResult
from foodly_move_msgs.msg import WaypointsAction
from foodly_move_msgs.msg import WaypointsGoal
from foodly_move_msgs.msg import WaypointsFeedback
from foodly_move_msgs.msg import WaypointsResult
from foodly_move_msgs.msg import MotionAction
from foodly_move_msgs.msg import MotionGoal
from foodly_move_msgs.msg import MotionResult
from foodly_move_msgs.msg import MotionFeedback
from foodly_move_msgs.srv import EefConfig, EefConfigRequest


sys.path.append('./src/foodly_package/detectron2')
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.structures.boxes import BoxMode
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import Instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer,VisImage
from sklearn.decomposition import PCA
import os


from std_msgs.msg import String, Int32

def calc_camdeg2pos( degree ):
    # function provided by RT, uesd for conversion of angle when giving end rotations to robot head
    CAMERA_BASE_X = 0.08195
    CAMERA_BASE_Z = 0.094
    NECK_BASE_Z = 0.474
    NECK_BASE_X = 0.105

    rad = -math.radians( degree )
    temp_x = CAMERA_BASE_X + 0.2
    temp_z = CAMERA_BASE_Z
    cam_x = ( temp_x * math.cos(rad) - temp_z * math.sin(rad) ) + NECK_BASE_X
    cam_z = ( temp_x * math.sin(rad) + temp_z * math.cos(rad) ) + NECK_BASE_Z
    return cam_x, cam_z


class MotionControl:
    def __init__(self):
        self.motion_client = actionlib.SimpleActionClient('/foodly/dev_api/move/motion', MotionAction)
        self.motion_client.wait_for_server()

        self.way_client = [ actionlib.SimpleActionClient('/foodly/dev_api/move/waypoints/right', WaypointsAction), \
                                actionlib.SimpleActionClient('/foodly/dev_api/move/waypoints/left', WaypointsAction) ]
        self.way_client[0].wait_for_server()
        self.way_client[1].wait_for_server()

        self.eef_client = [ actionlib.SimpleActionClient('/foodly/dev_api/move/endeffector/right', EndeffectorAction), \
                                actionlib.SimpleActionClient('/foodly/dev_api/move/endeffector/left', EndeffectorAction) ]
        self.eef_client[0].wait_for_server()
        self.eef_client[1].wait_for_server()

        self.camera_pub = rospy.Publisher('/foodly/dev_api/move/camera', Camera, queue_size=1)
        self.indicator_pub = rospy.Publisher('/foodly/dev_api/move/indicator', Indicator, queue_size=1)
        self.cancel_plan_pub = rospy.Publisher('/foodly/dev_api/move/cancel_plan', CancelPlan, queue_size=1)
        
        self.eef_config_srv = rospy.ServiceProxy('/foodly/dev_api/move/eefconfig', EefConfig)
        self.eef_config_srv.wait_for_service()
        
        # used to estimate time it will take arm to get into position
        self.ARM_TIME = rospy.Duration(3.0)
        
        # used to estimate time it takes arm to descend and grab object
        self.ARM_OFFSET = rospy.Duration(0.5)
        
        # used to estimate position of object in space on a conveyor belt
        self.OBJ_VELOCITY = 0.01
        
        self.prepare_arms()
        self.move_camera(75)
        self.indicator(138)


        rospy.sleep(2)
        print("Startup complete")
        
    def indicator(self, code):
        # Indicator Codes: 138 = Yellow blink, 73 = Green blink 
        indicator_msg = Indicator()
        indicator_msg.code = code
        self.indicator_pub.publish(indicator_msg)
    
    def home_position(self):
        # Home motion
        self.move_camera(0)
        motion_req = MotionGoal()
        motion_req.motion_id = 0
        self.motion_client.send_goal( motion_req )
        self.motion_client.wait_for_result(timeout=rospy.Duration(10.0))
    
    def prepare_arms(self):
        RIGHT_ARM = 0
        LEFT_ARM = 1
        EEF_LENGTH = 130    # [mm]
        FOOD_SIZE = 70      # [mm]
        CLOSE_SIZE = 60     # [mm]
        GRASP_TORQUE = 3.0  # [N]
        GRASP_THRESH = 2.0  # [N]
    
        self.home_position()
        
        # EndeffectorConfig
        eefconfig_req = EefConfigRequest()
        eefconfig_req.part = RIGHT_ARM
        eefconfig_req.eef_length = EEF_LENGTH
        eefconfig_req.food_size = FOOD_SIZE
        eefconfig_req.close_size = CLOSE_SIZE
        eefconfig_req.grasp_torque = GRASP_TORQUE
        eefconfig_req.grasp_threshold = GRASP_THRESH
        self.eef_config_srv(eefconfig_req)
    
        # EndeffectorConfig
        eefconfig_req = EefConfigRequest()
        eefconfig_req.part = LEFT_ARM
        eefconfig_req.eef_length = EEF_LENGTH
        eefconfig_req.food_size = FOOD_SIZE
        eefconfig_req.close_size = CLOSE_SIZE
        eefconfig_req.grasp_torque = GRASP_TORQUE
        eefconfig_req.grasp_threshold = GRASP_THRESH
        self.eef_config_srv(eefconfig_req)
    
        # Remove magnet (Right)
        motion_req = MotionGoal()
        motion_req.motion_id = 1
        self.motion_client.send_goal( motion_req )
        self.motion_client.wait_for_result(timeout=rospy.Duration(3.0))
        
        # Remove magnet (Left)
        motion_req = MotionGoal()
        motion_req.motion_id = 2
        self.motion_client.send_goal( motion_req )
        self.motion_client.wait_for_result(timeout=rospy.Duration(3.0))
        
    def move_camera(self, cam_angle):
        # Move camera
        cam_msg = Camera()
        cam_msg.position.y = 0.0
        cam_msg.position.x, cam_msg.position.z = calc_camdeg2pos(cam_angle)
        self.camera_pub.publish( cam_msg )
        
    def _create_way(self, eef_pos, eef_dir, eef_state):
        # Fill in all the data for each waypoint Pose
        way = Pose()
        way.position.x = eef_pos[0]
        way.position.y = eef_pos[1]
        way.position.z = eef_pos[2]
        way.direction.x = eef_dir[0]
        way.direction.y = eef_dir[1]
        way.direction.z = eef_dir[2]
        way.sec.data = rospy.Duration( 2.0 )
        way.eef = eef_state
        
        return way
        
    def intercept(self, arm, obj_info):
        EEF_OPEN = 0
        EEF_CLOSE = 1
        
        scan_time, eef_pos, eef_dir = obj_info
        
        start_time = rospy.get_rostime()
        y_offset = ((start_time - scan_time + self.ARM_TIME + self.ARM_OFFSET) * self.OBJ_VELOCITY).to_sec()
        
        # move to above intercept position
        waypoints = []
        eef_intercept_pos = eef_pos
        eef_intercept_pos[1] += y_offset
        eef_intercept_pos[2] += 0.1
        
        way = self._create_way(eef_intercept_pos, eef_dir, EEF_OPEN)

           
        waypoints.append( way )
        
        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )
        self.way_client[arm].wait_for_result( self.ARM_TIME )
        if not self.way_client[arm].get_state() == actionlib.GoalStatus.SUCCEEDED:
            return False
            
        while rospy.get_rostime() < start_time + self.ARM_TIME:
            pass
        
        # lower arm around object
        waypoints = []
        eef_intercept_pos[2] -= 0.05
        way = self._create_way(eef_intercept_pos, eef_dir, EEF_CLOSE)
            
        waypoints.append( way )
        
        # raise grasped object up
        eef_intercept_pos[2] += 0.05
        way = self._create_way(eef_intercept_pos, eef_dir, EEF_CLOSE)
        
        waypoints.append( way )
        
        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )
        self.way_client[arm].wait_for_result( self.ARM_TIME )
        
        return True
        
    def pick_place(self, arm, eef_pos, eef_dir, action, class_name):
        
        #self.Valve.serial_send("S", 1, True)
        if action != 1 and action != 0:
            print("Unrecognised Action")
            return

        
        
        EEF_OPEN = 0
        EEF_CLOSE = 1
        PICK = 0
        PLACE = 1


        if class_name == "Lime":
            eef_dir[2] = 90 - eef_dir[2]
            print (eef_dir)

        waypoints = []
        
        # move over produce

        eef_above_pos = eef_pos[:]
        eef__above_original_position = eef_above_pos[2]
        eef_above_pos[2] +=  0.2 # 0.1
        way = self._create_way(eef_above_pos, eef_dir, action)
        waypoints.append( way )





        #self.Valve.serial_send("S", 1, True)
        
        # lower pincer around produce
        eef_pos[2] += -0.09#-0.10 #-0.3 #Adjusting the Height
        way = self._create_way(eef_pos, eef_dir, action)
        waypoints.append( way )


        #action = (action + 1) % 2
        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )
        self.way_client[arm].wait_for_result( rospy.Duration.from_sec(5.0) )#20.0
        if self.way_client[arm].get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Aborted..")



        rospy.sleep(2)
        print("sleep")
        waypoints = []
        
        # move over produce

        eef_pos[2] +=  -0.04 # 0.1
        way = self._create_way(eef_pos, eef_dir, action)
        waypoints.append( way )

        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )
        self.way_client[arm].wait_for_result( rospy.Duration.from_sec(5.0) )#20.0
        if self.way_client[arm].get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Aborted..")



        # Decides the method of picking depending on the detected object
        if class_name == "Mango":
            print("Picking up {}".format(class_name))
            suction_off()
            SPA_off()
            
 
            suction_on()
            rospy.sleep(2)

        elif class_name == "Cucumber" or "Lime":
            print("Picking up {}".format(class_name))
            suction_off()
            SPA_off()
            

            SPA_on()
            rospy.sleep(3)
            suction_on()
            rospy.sleep(2)


        waypoints = []

        #rospy.sleep(10)
        way = self._create_way(eef_pos, eef_dir, action)
        waypoints.append( way )
        
        

        # raise above where produce was
        eef_above_pos = eef_pos[:]
        eef_above_pos[2] = eef__above_original_position
        way = self._create_way(eef_above_pos, eef_dir, action)
        waypoints.append( way )

        eef_pos[2] += 0.32#-0.10 #-0.3 #Adjusting the Height
        way = self._create_way(eef_pos, eef_dir, action)
        waypoints.append( way )



        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )
        self.way_client[arm].wait_for_result( rospy.Duration.from_sec(20.0) )#20.0
        if self.way_client[arm].get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Aborted..")
            return False
        return True

            
    def drop(self, arm, eef_pos):
        EEF_OPEN = 0
        EEF_CLOSE = 1
        waypoints = []
        eef_pos[2] += 0.2
        way = self._create_way(eef_pos, [0.0, 0.0, 0.0], EEF_OPEN)
        waypoints.append( way )
        
        action_goal = WaypointsGoal()
        action_goal.waypoint = waypoints
        action_goal.part = arm
        self.way_client[arm].send_goal( action_goal )

        #self.Valve.serial_send("S", 0, True)

        self.way_client[arm].wait_for_result( rospy.Duration.from_sec(10.0) )
        if self.way_client[arm].get_state() != actionlib.GoalStatus.SUCCEEDED:
            print("Aborted..")

        suction_off()
        SPA_off()
        rospy.sleep(3)


class ProcessImages:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_subscriber = rospy.Subscriber('camera/color/image_raw', Image, self.img_callback, queue_size = 1)
        self.registered_pc_subscriber = rospy.Subscriber('camera/depth_registered/points', PointCloud2, self.pc_callback, queue_size = 1)
        self.marker_publisher = rospy.Publisher("detected_centers", Marker, queue_size = 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.last_img = None
        self.last_pc = None
        self.produce_list = []
        self.id_counter = 0
        
        self.model = None
        self.load_model()
    
    def get_dicts(self, directory):
        classes = ["Mango", "Lime", "Cucumber"]  ### ADD ALL THE CLASSES LATER
        dataset_dicts = []
        #image_id = 0
        for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
            json_file = os.path.join(directory, filename)
            with open(json_file) as f:
                img_anns = json.load(f)

            record = {}
            #print(img_anns.keys())
            #print(img_anns["imageData"])
            filename = os.path.join(directory, img_anns["imagePath"])
        
            record["file_name"] = filename
            record["image_id"] = re.findall("\d+\.\d+", filename)[0]
            #print (re.findall("\d+\.\d+", filename)[0])
            #image_id += 1
            record["height"] = img_anns["imageHeight"]
            record["width"] = img_anns["imageWidth"]
      
            annos = img_anns["shapes"]
            objs = []
            for anno in annos:
                px = [a[0] for a in anno['points']]
                py = [a[1] for a in anno['points']]
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(anno['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts    


    def load_model(self):

        for d in ["train", "val"]:
            #register_coco_instances(d, {}, f"images/TMR/{d}/trainval.json", f"images/TMR/{d}")
            DatasetCatalog.register(d, lambda d=d: self.get_dicts("images/TMR/" + d))
            # Set the class name
            MetadataCatalog.get(d).set(thing_classes= ["Mango", "Lime", "Cucumber"])
        mango_meta_data = MetadataCatalog.get("train")
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #self.cfg.MODEL.WEIGHTS = "./src/foodly_felix/detectron2/output/model_final.pth" # path for final model
        self.cfg.MODEL.WEIGHTS = os.path.join(os.path.expanduser('~'), 'catkin_ws', 'src', 'foodly_felix', 'detectron2', 'output', 'model_final_new.pth')
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  ### UPDATE THIS TOO
        self.cfg.MODEL.DEVICE = "cpu"

        

        
    def img_callback(self, data):
        #predictor = DefaultPredictor(self.cfg)
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imshow("input image", img)
            # convert bgr image to rgb
            self.last_img = img[:, :, ::-1].copy()
            key = cv2.waitKey(1)
        except CvBridgeError as e:
            print(e)
            
    def pc_callback(self, data):
        self.last_pc = data
   
    def create_marker(self, x, y, z):
        pos = PointStamped()
        pos.point.x = x
        pos.point.y = y
        pos.point.z = z

        obj_marker = Marker()
        obj_marker.header.frame_id = "base_link"
        obj_marker.header.stamp    = rospy.get_rostime()
        obj_marker.ns = "produce"
        obj_marker.id = self.id_counter
        self.id_counter += 1
        
        obj_marker.type = 2 # sphere
        obj_marker.action = 0
        obj_marker.pose.position = pos.point
        obj_marker.pose.orientation.x = 0
        obj_marker.pose.orientation.y = 0
        obj_marker.pose.orientation.z = 0
        obj_marker.pose.orientation.w = 1.0
        obj_marker.scale.x = 0.01
        obj_marker.scale.y = 0.01
        obj_marker.scale.z = 0.01

        obj_marker.color.r = 0.0
        obj_marker.color.g = 1.0
        obj_marker.color.b = 0.0
        obj_marker.color.a = 1.0

        obj_marker.lifetime = rospy.Duration(0)
        
        self.marker_publisher.publish(obj_marker)

    def find_produce(self):
        if 1==1:#self.model is not None and self.last_pc is not None and self.last_img is not None:
            print("entered find produces")
            start_time = rospy.get_rostime()
            try:
                trans = self.tf_buffer.lookup_transform("base_link", self.last_pc.header.frame_id, self.last_pc.header.stamp, rospy.Duration(20.0))
            except tf2.LookupException as ex:
                rospy.logwarn(ex)
                return
            except tf2.LookupException as ex:
                rospy.logwarn(ex)
                return
                
            trans_cloud = do_transform_cloud(self.last_pc, trans)
            pc_gen = pc2.read_points(trans_cloud, skip_nans=False)
            np_scene = np.zeros((360, 640, 3), dtype=np.float32)
            
            for i, point in enumerate(pc_gen):
                x, y, z, _ = point
                if not np.isnan(x):
                    np_scene[i // 640, i % 640] = x, y, z
                    
            output_img = self.last_img.copy()

            #Converted to the correct colour space
            output_img2 = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            #self.cfg.MODEL.WEIGHTS = "./src/foodly_package/detectron2/output/model_final.pth" # path for final model
            #self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

            
            predictor = DefaultPredictor(self.cfg)
            #output: Instances = predictor(output_img2)["instances"]
            output = predictor(output_img2)
            mask_instances = output["instances"]


            detected_class_indexes = mask_instances.pred_classes
            metadata = MetadataCatalog.get("train")
            class_catalog = metadata.thing_classes


            v = Visualizer(output_img2,
                        MetadataCatalog.get("val"),
                        scale = 1.0)
            result: VisImage = v.draw_instance_predictions(mask_instances.to("cpu"))
            result_image: np.ndarray = result.get_image()[..., ::-1][..., ::-1]
            
            time_now = str(time.time()) 

            out_file_name = "image/masked_image_"
            out_file_name += time_now
            out_file_name += ".png"
            print(out_file_name)
            #print (mask_instances)
            cv2.imshow(out_file_name, result_image)
            cv2.imwrite(out_file_name, result_image)


            masks = mask_instances.pred_masks.to("cpu")#.tolist() (N, H, W)
            masks = masks.numpy()
            masks = np.transpose(masks, (1, 2, 0))

            for mask_idx in range(masks.shape[-1]):
                # draw mask prediction from Mask R-CNN on output image
                class_index = detected_class_indexes[mask_idx]
                class_name = class_catalog[class_index]
            
                #output_img2 = np.uint8(np.where(masks[:, :, mask_idx, None], np.random.randint(256, size=3), output_img2))  

                # associate pixels in mask to their 3D positions (N, 3)
                points = np.where(masks[:, :, mask_idx])
                points_space = np_scene[points]
                points_space = points_space[~np.all(points_space == 0, axis=1)]
                points_space = points_space[np.all((-0.6 < points_space) & (points_space < 0.6), axis=1)]
                
                # find minimum bounding box for each mask on rgb image
                points_pairs = np.column_stack(points)
                rect_img = cv2.minAreaRect(points_pairs[:, ::-1])
                (x, y), (w, h), angle = rect_img
                x = int(x)
                y = int(y)
                # print(angle, w, h)
                if w > h:
                    angle += 90
                
                # draw minimum bounding box on output image
                box = cv2.boxPoints(rect_img)
                box = np.int32(box)
                cv2.drawContours(output_img, [box], 0, (0, 255, 0), 2)
                cv2.circle(output_img, (x, y), 3, (0, 0, 255), 1)
                
                # find minimum bounding box for points in space
                
            



                #bounding box corners (of the largest potato)
                #print (masks[mask_idx].pred_boxes)
                x_l = list(output["instances"].to("cpu").pred_boxes)[mask_idx][0].numpy()
                y_l = list(output["instances"].to("cpu").pred_boxes)[mask_idx][1].numpy()
                x_r = list(output["instances"].to("cpu").pred_boxes)[mask_idx][2].numpy()
                y_r = list(output["instances"].to("cpu").pred_boxes)[mask_idx][3].numpy()



                #Obtain the boolean mask (true or false)
                mask_whole = list(output["instances"].to("cpu").pred_masks)[mask_idx].numpy()
                imask = mask_whole[int(y_l):int(y_r),int(x_l):int(x_r)] 

                #bit mask in 1s and zeros
                mask = imask * 1

                #For displaying the mask using cv2
                #disp_mask = (mask*255).astype(np.uint8)

                #PCA for orientation detection

                #Puts segmented mask at centre
                ypc, xpc = int((y_r - y_l)*0.5)+1, int((x_r - x_l)*0.5)+1

                #Indices of where the mask has a value of 1 (turn image ata into data points)
                one_indices = np.transpose(np.asarray(np.where(mask == 1)))

                #Get the coordinates wrt a centre axes
                one_indices[:, 0] = ypc - one_indices[:, 0]
                one_indices[:, 1] = one_indices[:, 1] - xpc
                #Convert to the correct format for the data
                y_data = one_indices[:,0]
                x_data = one_indices[:,1]
                data = np.zeros_like(one_indices)
                data[:,0] = x_data
                data[:,1] = y_data

                #Gets the main principle component
                pca = PCA(n_components=1).fit(data)

                for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
                    #print(comp[1],comp[0])
                    angle = np.arctan2(comp[1],comp[0])
                    
                    #Constrain angle between -90 and 90 degrees
                    if abs(angle) > np.pi:
                        angle %= (2*np.pi)

                    if angle > np.pi/2.0:
                        angle = angle - np.pi
                    if angle < -np.pi/2.0:
                        angle = np.pi + angle


                angle_degrees = angle*180/np.pi
                if angle_degrees < 0 and angle > -80:
                    angle_degrees = -90 -angle_degrees
                elif angle_degrees >= 0 and angle < 80:

                    #this works
                    angle_degrees = 90 - angle_degrees


                print("after", angle_degrees)
                
                

                width = x_r - x_l
                height = y_r - y_l
                #Bounding box centre wrt to the actual image
                x_centre = int(x_l + width/2)
                y_centre = int(y_l + height/2)

                if len(points_space) > 0:
                    scene_x = np.average(points_space, axis = 0)[0]
                    scene_y = np.average(points_space, axis = 0)[1]
                    scene_z = np.average(points_space, axis = 0)[2]
                    print(scene_x)
                    print(scene_y)
                    print(scene_z)


                    if scene_z < 0.2:
                        self.produce_list.append((start_time, [scene_x, scene_y, scene_z], [0, -0.5, -angle_degrees], class_name))
                        self.create_marker(scene_x, scene_y, scene_z)




            cv2.imshow("processed image", output_img[:, :, ::-1])
            out_file_name = "image/processed_image_"
            out_file_name += time_now
            out_file_name += ".png"
            cv2.imwrite(out_file_name, output_img[:, :, ::-1])

##################

########################################


## Publisher for proportional valve controller
#!/usr/bin/env python3
def suction_on():
    pub = rospy.Publisher('chatter', Int32, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    pub.publish(11) #turn Suction on
    print("Suction on")


def suction_off():
    pub = rospy.Publisher('chatter', Int32, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    pub.publish(10) #turn Suction off
    print("Suction off")

def SPA_on():
    pub = rospy.Publisher('chatter', Int32, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    pub.publish(21) #turn SPA on
    print("SPA on")

def SPA_off():
    pub = rospy.Publisher('chatter', Int32, queue_size=10)
    rate = rospy.Rate(10) # 10hz 
    pub.publish(20) #turn SPA off
    print("SPA off")





if __name__ == '__main__':
    EEF_OPEN = 0
    EEF_CLOSE = 1
    RIGHT_ARM = 0
    LEFT_ARM = 1
    PICK = 0
    PLACE = 1

    

    rospy.init_node('picker')

    image_processor = ProcessImages()
    motion_controller = MotionControl()
    motion_controller.drop(RIGHT_ARM, [0.45, -0.3, 0.1])
    motion_controller.drop(LEFT_ARM, [0.45, 0.3, 0.1])

    while image_processor.produce_list == []:
        image_processor.find_produce()

    

    for obj_info in image_processor.produce_list:
        start, eef_pos, eef_dir, class_name = obj_info
        #eef_pos[3] -= 0.01
        if eef_pos[2] < -0.2:
            print("bad")
            eef_pos[2] = -0.2
            
        if eef_pos[1] <= 0.0:            
            if not motion_controller.pick_place(RIGHT_ARM, eef_pos, eef_dir, PICK, class_name):
                continue
            motion_controller.drop(RIGHT_ARM, [0.45, -0.3, 0.0])

        else:
            if not motion_controller.pick_place(LEFT_ARM, eef_pos, eef_dir, PICK, class_name):
                continue
            motion_controller.drop(LEFT_ARM, [0.45, 0.3, 0.0])

    
            
        
        
    motion_controller.home_position()
    motion_controller.indicator(73)
    

    cv2.destroyAllWindows()





