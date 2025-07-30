import cv2
import os
import numpy as np
import torch
import time
import argparse
from Sim3DR.renderer import render_fvr
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import onnxruntime as ort
from faceversev4 import FaceVerseModel_torch

stop_flag = False

def distance(x, y):
    return np.sqrt(((x - y) ** 2).sum())

def norm(x):
    return x / np.sqrt((x ** 2).sum())

class ONNXFaceVerseInference:
    def __init__(self, faceverse_path, onnx_path, device='cuda'):
        """
        ONNX-based FaceVerse inference class
        """
        self.device = device
        
        # Load ONNX model
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Load FaceVerse model for rendering
        self.faceverse_model = FaceVerseModel_torch(
            device=torch.device(device),
            facevrsepath=faceverse_path,
            camera_distance=10,
            focal=1000,
            center=128
        )
        
        print(f"✅ ONNX model loaded: {onnx_path}")
        print(f"Provider: {self.ort_session.get_providers()}")
        
        if 'CUDAExecutionProvider' in self.ort_session.get_providers():
            print("✅ Running on GPU")
        else:
            print("⚠️ Running on CPU")
    
    def preprocess_image(self, image, bbox):
        """
        Preprocess image for ONNX model
        """
        # Crop face
        x1, y1, x2, y2 = bbox.astype(int)
        face_crop = image[y1:y2, x1:x2]
        
        # Resize to 256x256
        face_resized = cv2.resize(face_crop, (256, 256))
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert to CHW format
        face_chw = np.transpose(face_normalized, (2, 0, 1))
        
        # Add batch dimension
        face_batch = np.expand_dims(face_chw, axis=0)
        
        return face_batch
    
    def predict_parameters(self, image, bbox):
        """
        Predict parameters using ONNX model
        """
        # Preprocessing
        input_tensor = self.preprocess_image(image, bbox)
        
        # ONNX inference
        start_time = time.time()
        outputs = self.ort_session.run(['output'], {'input': input_tensor})
        inference_time = time.time() - start_time
        
        # Return 621-dimensional parameters
        parameters = outputs[0][0]  # [621]
        
        return parameters, inference_time
    
    def from_coeffs(self, coeffs, bbox_list):
        """
        Generate 3D face from coefficients (same as original)
        """
        # Convert to PyTorch tensor
        coeffs_tensor = torch.from_numpy(coeffs).float().to(self.device)
        
        # Generate 3D face
        with torch.no_grad():
            result = self.faceverse_model.run(coeffs_tensor, use_color=True)
            
            # Handle dict return type
            if isinstance(result, dict):
                vertices = result.get('vertices', None)
                colors = result.get('colors', None)
                faces = result.get('faces', None)
            else:
                vertices, colors, faces = result, None, None
            
            # Project vertices - fix tensor dimensions
            vs_proj = vertices.clone()
            
            # Debug bbox_list shape (commented out)
            # print(f"bbox_list shape: {bbox_list.shape}, dim: {bbox_list.dim()}")
            
            # Extract bbox dimensions properly
            if bbox_list.dim() == 3:  # [batch, 1, 4] format
                bbox_list = bbox_list.squeeze(1)  # Remove middle dimension
                bbox_width = bbox_list[:, 2] - bbox_list[:, 0]  # width
                bbox_height = bbox_list[:, 3] - bbox_list[:, 1]  # height
                bbox_center_x = (bbox_list[:, 0] + bbox_list[:, 2]) / 2  # center x
                bbox_center_y = (bbox_list[:, 1] + bbox_list[:, 3]) / 2  # center y
                
                # Project vertices
                vs_proj[:, :, 0] = vs_proj[:, :, 0] * bbox_width.unsqueeze(1) + bbox_center_x.unsqueeze(1)
                vs_proj[:, :, 1] = vs_proj[:, :, 1] * bbox_height.unsqueeze(1) + bbox_center_y.unsqueeze(1)
            elif bbox_list.dim() == 2:  # [batch, 4] format
                bbox_width = bbox_list[:, 2] - bbox_list[:, 0]  # width
                bbox_height = bbox_list[:, 3] - bbox_list[:, 1]  # height
                bbox_center_x = (bbox_list[:, 0] + bbox_list[:, 2]) / 2  # center x
                bbox_center_y = (bbox_list[:, 1] + bbox_list[:, 3]) / 2  # center y
                
                # Project vertices
                vs_proj[:, :, 0] = vs_proj[:, :, 0] * bbox_width.unsqueeze(1) + bbox_center_x.unsqueeze(1)
                vs_proj[:, :, 1] = vs_proj[:, :, 1] * bbox_height.unsqueeze(1) + bbox_center_y.unsqueeze(1)
            elif bbox_list.dim() == 1 and bbox_list.size(0) == 4:  # [4] format
                # Single bbox
                bbox_width = bbox_list[2] - bbox_list[0]
                bbox_height = bbox_list[3] - bbox_list[1]
                bbox_center_x = (bbox_list[0] + bbox_list[2]) / 2
                bbox_center_y = (bbox_list[1] + bbox_list[3]) / 2
                
                vs_proj[:, :, 0] = vs_proj[:, :, 0] * bbox_width + bbox_center_x
                vs_proj[:, :, 1] = vs_proj[:, :, 1] * bbox_height + bbox_center_y
            else:
                # Handle other cases - use default values
                print(f"Warning: Unexpected bbox_list shape: {bbox_list.shape}")
                # Use default projection (no scaling)
                pass
            
            # Calculate normals
            normal = torch.zeros_like(vertices)
            # Simple normal calculation (you might want to implement proper normal calculation)
            
            return vertices, vs_proj, normal, colors

class FrameLoader(threading.Thread):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mode = self._determine_mode()

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        running_mode = VisionRunningMode.VIDEO if self.mode == 'video' or self.mode == 'webcam' else VisionRunningMode.IMAGE
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='data/face_landmarker.task'),
            running_mode=running_mode,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        self.face_tracker = vision.FaceLandmarker.create_from_options(options)
        
        self.cap = None
        if self.mode == 'video':
            self.cap = cv2.VideoCapture(self.args.input)
        elif self.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
        self.current_index = 0
        self.frames_queue = []
        self.boxes_queue = []
        self.frame_names_queue = []
        self.done = False
        self.max_queue_size = 3

    def detect_lms(self, img):
        face_image = mp.Image(mp.ImageFormat.SRGB, img.astype(np.uint8))
        if self.mode == 'video' or self.mode == 'webcam':
            results = self.face_tracker.detect_for_video(face_image, self.current_index * 33)
        else:
            results = self.face_tracker.detect(face_image)
        if not results.face_landmarks:
            return []
        else:
            lms = results.face_landmarks[0]
            lms = np.array([(lmk.x, lmk.y) for lmk in lms])
            lms[:, 0] = lms[:, 0] * img.shape[1]
            lms[:, 1] = lms[:, 1] * img.shape[0]
            vec_norm = norm(lms[362, :2] - lms[263, :2])
            leyex = np.dot(lms[473] - (lms[263, :2] + lms[362, :2]) / 2, vec_norm) / distance(lms[362, :2], lms[263, :2]) * 3
            leyey = np.dot(lms[473] - (lms[263, :2] + lms[362, :2]) / 2, vec_norm[[1, 0]]) / distance(lms[362, :2], lms[263, :2]) * -1.5
            vec_norm = norm(lms[33, :2] - lms[133, :2])
            reyex = np.dot(lms[468] - (lms[33, :2] + lms[133, :2]) / 2, vec_norm) / distance(lms[33, :2], lms[133, :2]) * 3
            reyey = np.dot(lms[468] - (lms[33, :2] + lms[133, :2]) / 2, vec_norm[[1, 0]]) / distance(lms[33, :2], lms[133, :2]) * -1.5
        return [[np.min(lms[:, 0]), np.min(lms[:, 1]), np.max(lms[:, 0]), np.max(lms[:, 1])], [leyey, leyex, reyey, reyex]]

    def _determine_mode(self):
        if self.args.input.endswith('.mp4') or self.args.input.endswith('.MP4') or self.args.input.endswith('.avi'):
            return 'video'
        elif self.args.input.endswith('.png') or self.args.input.endswith('.jpg') or self.args.input.endswith('.jpeg'):
            return 'image'
        elif self.args.input.endswith('webcam'):
            return 'webcam'
        return 'imagefolder'

    def run(self):
        if self.mode == 'video' or self.mode == 'webcam':
            while True:
                if stop_flag or self.done:
                    break
                if len(self.frames_queue) < self.max_queue_size:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_name = f"frame_{str(self.current_index).zfill(5)}.jpg"
                        self.current_index += 1
                        self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                        self.frames_queue.append(frame[:, :, ::-1])
                        self.frame_names_queue.append(frame_name)
                    else:
                        self.cap.release()
                        self.done = True
                        break
                else:
                    time.sleep(0.01)
                    continue
        elif self.mode == 'image':
            frame = cv2.imread(self.args.input)[:, :, :3]
            file_name = os.path.basename(self.args.input)
            if frame is not None:
                self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                self.frames_queue.append(frame[:, :, ::-1])
                self.frame_names_queue.append(file_name)
            self.done = True
        elif self.mode == 'imagefolder':
            root, dirs, files = next(os.walk(self.args.input))
            for file in files:
                if stop_flag or self.done:
                    break
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    file_path = os.path.join(root, file)
                    frame = cv2.imread(file_path)[:, :, :3]
                    if frame is not None:
                        if len(self.frames_queue) < self.max_queue_size:
                            self.boxes_queue.append(self.detect_lms(frame[:, :, ::-1]))
                            self.frames_queue.append(frame[:, :, ::-1])
                            self.frame_names_queue.append(file)
                        else:
                            time.sleep(0.01)
                            continue
            self.done = True

    def get_next_frame(self):
        if len(self.frames_queue) > 0:
            frame = self.frames_queue.pop(0)
            frame_name = self.frame_names_queue.pop(0)
            boxes = self.boxes_queue.pop(0)
            return frame, frame_name, boxes
        return None, None, None

def ply_from_array_color(points, colors, faces, output_file):
    num_points = len(points)
    num_triangles = len(faces)

    header = '''ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face {}
property list uchar int vertex_indices
end_header\n'''.format(num_points, num_triangles)

    with open(output_file,'w') as f:
        f.writelines(header)
        index = 0
        for item in points:
            f.write("{0:0.6f} {1:0.6f} {2:0.6f} {3} {4} {5}\n".format(item[0], item[1], item[2],
                                                        colors[index, 0], colors[index, 1], colors[index, 2]))
            index = index + 1

        for item in faces:
            number = len(item)
            row = "{0}".format(number)
            for elem in item:
                row += " {0} ".format(elem)
            row += "\n"
            f.write(row)

def process_one_batch_onnx(args, frameloader, onnx_inference, box_batch, frame_batch, name_batch, cache_frame, cache_param, video_writer=None, local_stop_flag=None, end=False):
    if len(name_batch) > 0:
        box_batch = np.stack(box_batch)
        eye_batch = box_batch[:, 1]
        box_batch = box_batch[:, 0:1]
        frame_batch = np.stack(frame_batch)

        # Process each frame individually with ONNX
        coeffs_list = []
        inference_times = []
        
        for i, (frame, box) in enumerate(zip(frame_batch, box_batch)):
            parameters, inference_time = onnx_inference.predict_parameters(frame, box[0])
            coeffs_list.append(parameters)
            inference_times.append(inference_time)
        
        coeffs = np.stack(coeffs_list)
        # Force to get pupil pos from mediapipe
        coeffs[:, -4:] = eye_batch

    if args.visual:
        for index, name in enumerate(name_batch):
            if args.smooth:
                if len(cache_param) == 1:
                    coeffs_this = cache_param[-1]['coeffs']
                    bbox_list_this = cache_param[-1]['bbox_list']
                    framet_this = cache_frame[-1]['frame']
                    namet_this = cache_frame[-1]['name']
                elif len(cache_param) == 2:
                    coeffs_this = (cache_param[-2]['coeffs'] + cache_param[-1]['coeffs'] + coeffs[index:index+1]) / 3
                    bbox_list_this = cache_param[-1]['bbox_list']
                    framet_this = cache_frame[-1]['frame']
                    namet_this = cache_frame[-1]['name']
                else:
                    coeffs_this = None
            else:
                coeffs_this = coeffs[index:index+1]
                bbox_list_this = box_batch[index:index+1, 0]  # Take only the bbox, not eyes
                framet_this = frame_batch[index]
                namet_this = name

            if coeffs_this is not None:
                vs, vs_proj, normal, colors = onnx_inference.from_coeffs(coeffs_this, torch.from_numpy(bbox_list_this).to(onnx_inference.device))
                # Convert tensors to numpy for render_fvr
                vs_proj_np = vs_proj[0].cpu().numpy()
                normal_np = normal[0].cpu().numpy()
                colors_np = colors[0].cpu().numpy()
                rgb, depth = render_fvr(framet_this, vs_proj_np, onnx_inference.faceverse_model.fvd["tri"], normal_np, colors_np)

                # Resize 3D render to match original frame size
                if rgb.shape[:2] != framet_this.shape[:2]:
                    rgb_resized = cv2.resize(rgb, (framet_this.shape[1], framet_this.shape[0]))
                else:
                    rgb_resized = rgb

                save_img = np.concatenate([framet_this[:, :, ::-1], rgb_resized[:, :, ::-1]], axis=0)
                bbox_list_this = bbox_list_this.astype(np.int32)
                
                # Handle different bbox shapes
                if bbox_list_this.ndim == 3:  # [1, 1, 4]
                    bbox = bbox_list_this[0, 0]
                elif bbox_list_this.ndim == 2:  # [1, 4]
                    bbox = bbox_list_this[0]
                else:  # [4]
                    bbox = bbox_list_this
                
                cv2.rectangle(save_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Add inference time info
                if index < len(inference_times):
                    cv2.putText(save_img, f"ONNX: {inference_times[index]*1000:.1f}ms", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save video frame if video writer is available
                if video_writer is not None:
                    video_writer.write(save_img)
                
                if args.save_results:
                    cv2.imwrite(os.path.join(args.output, namet_this), save_img)
                    headparam = {'coeffs': coeffs_this, 'bbox_list': bbox_list_this}
                    np.save(os.path.join(args.output, namet_this.split('.')[0]), headparam)
                    if args.save_ply:
                        ply_from_array_color(vs[0].cpu().numpy(), (colors[0] * 255).cpu().numpy().astype(np.uint8), 
                                           onnx_inference.faceverse_model.fvd["tri"], 
                                           os.path.join(args.output, namet_this.split('.')[0] + '.ply'))

                if frameloader.mode == 'webcam':
                    cv2.imshow('Webcam (ONNX)', save_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("🛑 'Q' pressed - stopping webcam...")
                        frameloader.done = True
                        if local_stop_flag is not None:
                            local_stop_flag[0] = True
                        stop_flag = True
                        break
    
            if args.smooth:
                headparam = {'coeffs': coeffs[index:index+1], 'bbox_list': box_batch[index:index+1]}
                cache_param.append(headparam)
                frame_t = {'frame': frame_batch[index], 'name': name}
                cache_frame.append(frame_t)
                if len(cache_param) > 2:
                    cache_param.pop(0)
                    cache_frame.pop(0)
        
        if end:
            while len(cache_param) > 1:
                coeffs_this = cache_param[-1]['coeffs']
                bbox_list_this = cache_param[-1]['bbox_list']
                framet_this = cache_frame[-1]['frame']
                namet_this = cache_frame[-1]['name']

                vs, vs_proj, normal, colors = onnx_inference.from_coeffs(coeffs_this, torch.from_numpy(bbox_list_this).to(onnx_inference.device))
                # Convert tensors to numpy for render_fvr
                vs_proj_np = vs_proj[0].cpu().numpy()
                normal_np = normal[0].cpu().numpy()
                colors_np = colors[0].cpu().numpy()
                rgb, depth = render_fvr(framet_this, vs_proj_np, onnx_inference.faceverse_model.fvd["tri"], normal_np, colors_np)

                # Resize 3D render to match original frame size
                if rgb.shape[:2] != framet_this.shape[:2]:
                    rgb_resized = cv2.resize(rgb, (framet_this.shape[1], framet_this.shape[0]))
                else:
                    rgb_resized = rgb

                save_img = np.concatenate([framet_this[:, :, ::-1], rgb_resized[:, :, ::-1]], axis=0)
                bbox_list_this = bbox_list_this.astype(np.int32)
                
                # Handle different bbox shapes
                if bbox_list_this.ndim == 3:  # [1, 1, 4]
                    bbox = bbox_list_this[0, 0]
                elif bbox_list_this.ndim == 2:  # [1, 4]
                    bbox = bbox_list_this[0]
                else:  # [4]
                    bbox = bbox_list_this
                
                cv2.rectangle(save_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Save video frame if video writer is available
                if video_writer is not None:
                    video_writer.write(save_img)
                
                if args.save_results:
                    cv2.imwrite(os.path.join(args.output, namet_this), save_img)
                    headparam = {'coeffs': coeffs_this, 'bbox_list': bbox_list_this}
                    np.save(os.path.join(args.output, namet_this.split('.')[0]), headparam)
                    if args.save_ply:
                        ply_from_array_color(vs[0].cpu().numpy(), (colors[0] * 255).cpu().numpy().astype(np.uint8), 
                                           onnx_inference.faceverse_model.fvd["tri"], 
                                           os.path.join(args.output, namet_this.split('.')[0] + '.ply'))

                if frameloader.mode == 'webcam':
                    cv2.imshow('Webcam (ONNX)', save_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("🛑 'Q' pressed - stopping webcam...")
                        frameloader.done = True
                        if local_stop_flag is not None:
                            local_stop_flag[0] = True
                        stop_flag = True
                        break
                cache_param.pop(0)
                cache_frame.pop(0)
            while len(cache_param) > 0:
                cache_param.pop(0)
                cache_frame.pop(0)

def run_onnx(args, onnx_inference):
    global stop_flag
    frameloader = FrameLoader(args)
    frameloader.start()

    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    if args.save_video:
        os.makedirs(args.video_output, exist_ok=True)
        print(f"📁 Video output folder: {args.video_output}")
    if args.save_results:
        os.makedirs(args.frames_output, exist_ok=True)
        print(f"📁 Frames output folder: {args.frames_output}")

    # Initialize video writer if save_video is enabled
    video_writer = None
    if args.save_video:
        # Get video properties from first frame
        first_frame = None
        while first_frame is None:
            frame, frame_name, boxes = frameloader.get_next_frame()
            if frame is not None:
                first_frame = frame
                break
            time.sleep(0.01)
        
        if first_frame is not None:
            height, width = first_frame.shape[:2]
            # Double height for concatenated image (original + 3D reconstruction)
            output_height = height * 2
            output_width = width
            
            # Create video writer in separate folder
            if frameloader.mode == 'webcam':
                output_video_path = os.path.join(args.video_output, "webcam_output.mp4")
                print(f"📹 Webcam video will be saved to: {output_video_path}")
            else:
                output_video_path = os.path.join(args.video_output, "output_video.mp4")
                print(f"🎥 Video will be saved to: {output_video_path}")
            
            # Use H.264 codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (output_width, output_height))
            print(f"📐 Video size: {output_width}x{output_height}")
            print(f"🎬 FPS: 30")

    if frameloader.mode == 'webcam':
        args.batch = 1

    # for smooth  
    box_buffer = []
    frame_buffer = []
    name_buffer = []

    # for network input
    box_batch = []
    frame_batch = []
    name_batch = []

    frame_num = 0
    start_time_total = time.time()

    cache_frame = []
    cache_param = []
    
    # Local stop flag for this function
    local_stop_flag = [False]

    while True:
        if local_stop_flag[0]:
            print("🛑 Stopping due to stop flag...")
            break
        frame, frame_name, boxes = frameloader.get_next_frame()
        if frame is None:
            if frameloader.done:
                break
            time.sleep(0.01)
            continue
        
        if len(boxes) > 0:
            box = np.array(boxes).astype(np.float32)[0, :4]
            eyes = np.array(boxes).astype(np.float32)[1, :4]

            width = box[2] - box[0]
            height = box[3] - box[1]
            side_length = max(width, height)
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            box[0] = center_x - side_length // 2
            box[1] = center_y - side_length // 2
            box[2] = center_x + side_length // 2
            box[3] = center_y + side_length // 2

            if args.smooth:
                box_buffer.append(np.stack([box, eyes]))
                frame_buffer.append(frame)
                name_buffer.append(frame_name)
                if len(box_buffer) > 3:
                    box_buffer.pop(0)
                    frame_buffer.pop(0)
                    name_buffer.pop(0)
                smoothed_box = np.mean(box_buffer, axis=0) if len(box_buffer) == 3 else box_buffer[0]
                frame_this = frame_buffer[1] if len(box_buffer) > 1 else frame_buffer[0]
                name_this = name_buffer[1] if len(box_buffer) > 1 else name_buffer[0]
            else:
                smoothed_box = np.stack([box, eyes])
                frame_this = frame
                name_this = frame_name

            if len(box_batch) < args.batch:
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
            else:
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
                process_one_batch_onnx(args, frameloader, onnx_inference, box_batch, frame_batch, name_batch, cache_frame, cache_param, video_writer, local_stop_flag)
                box_batch = []
                frame_batch = []
                name_batch = []
        else:
            if args.smooth and len(box_buffer) > 0:
                smoothed_box = box_buffer[-1]
                frame_this = frame_buffer[-1]
                name_this = name_buffer[-1]
                box_batch.append(smoothed_box)
                frame_batch.append(frame_this)
                name_batch.append(name_this)
                process_one_batch_onnx(args, frameloader, onnx_inference, box_batch, frame_batch, name_batch, cache_frame, cache_param, video_writer, local_stop_flag, end=True)
                box_batch = []
                frame_batch = []
                name_batch = []
            box_buffer = []
            frame_buffer = []
            name_buffer = []
            print("No face detected in", frame_name)
            if frameloader.mode == 'webcam':
                save_img = np.concatenate([frame[:, :, ::-1], np.zeros_like(frame)], axis=0)
                cv2.imshow('Webcam (ONNX)', save_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("🛑 'Q' pressed - stopping webcam...")
                    frameloader.done = True
                    local_stop_flag[0] = True
                    stop_flag = True
                    break
            continue

        frame_num += 1
        frame_time = (time.time() - start_time_total) / frame_num
        if frame_num % 100 == 0:
            print(f"Processing time for frame {frame_name}: {frame_time:.4f} seconds", frame.shape)

    # process the last one frame
    if args.smooth and len(box_buffer) > 0:
        smoothed_box = box_buffer[-1]
        frame_this = frame_buffer[-1]
        name_this = name_buffer[-1]
        box_batch.append(smoothed_box)
        frame_batch.append(frame_this)
        name_batch.append(name_this)

    process_one_batch_onnx(args, frameloader, onnx_inference, box_batch, frame_batch, name_batch, cache_frame, cache_param, video_writer, local_stop_flag, end=True)

    frameloader.join()
    
    # Close video writer if it exists
    if video_writer is not None:
        try:
            video_writer.release()
            print("🎥 Video saved successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not close video writer properly: {e}")
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    
    end_time_total = time.time()
    total_time = end_time_total - start_time_total
    print(f"Total running time: {total_time:.4f} seconds")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="example/input/test.mp4")  # video or image or imagefolder or 'webcam'
    parser.add_argument("--output", type=str, default="example/output")
    parser.add_argument("--video_output", type=str, default="example/video_output", help="Separate folder for video output")
    parser.add_argument("--frames_output", type=str, default="example/frames_output", help="Separate folder for frame images")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--smooth", type=bool, default=True)
    parser.add_argument("--save_results", type=bool, default=False)
    parser.add_argument("--save_ply", type=bool, default=False)
    parser.add_argument("--save_video", type=bool, default=False, help="Save output as video")
    parser.add_argument("--visual", type=bool, default=True)
    parser.add_argument("--onnx_model", type=str, default="data/faceverse_resnet50_float32.onnx", help="ONNX model path")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load ONNX inference
    onnx_inference = ONNXFaceVerseInference(
        faceverse_path="data/faceverse_v4_2.npy",
        onnx_path=args.onnx_model,
        device='cuda'  # Force GPU usage
    )

    try:
        run_onnx(args, onnx_inference)
    except KeyboardInterrupt:
        stop_flag = True
        print("🛑 KeyboardInterrupt - stopping script...")
        print("Running script has been killed!") 