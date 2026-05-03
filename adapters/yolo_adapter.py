import torch
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
import os

class YOLOv8Adapter:
    """
    Adapter to connect YOLOv8 framework to the Advanced Analytics Core.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        
    def get_predictions(self, img_path):
        """
        Extract confidence scores, bboxes, and class IDs from an image.
        """
        results = self.model.predict(img_path, verbose=False)
        if not results:
            return [], [], []
            
        res = results[0]
        confs = res.boxes.conf.cpu().numpy()
        bboxes = res.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        clss = res.boxes.cls.cpu().numpy()
        return confs, bboxes, clss

    def generate_gradcam(self, img_path, target_layer_idx=-2):
        """
        Generate Grad-CAM heatmap for the given image.
        """
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = F.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(self.model.device)
        
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
            
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
            
        try:
            target_layer = self.model.model.model[target_layer_idx]
            f_hook = target_layer.register_forward_hook(forward_hook)
            b_hook = target_layer.register_full_backward_hook(backward_hook)
            
            # Ensure gradients are enabled and avoid inference mode
            with torch.set_grad_enabled(True):
                # Clone tensor to avoid "Inference tensors cannot be saved for backward"
                input_tensor = img_tensor.clone().detach()
                input_tensor.requires_grad = True
                preds = self.model.model(input_tensor)
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                
                # Extract top score
                scores = preds[0, 4:, :]
                max_score, _ = torch.max(scores.flatten(), dim=0)
                
                # Backward pass
                self.model.model.zero_grad()
                max_score.backward(retain_graph=True)
            
            f_hook.remove()
            b_hook.remove()
            
            if not activations or not gradients:
                return None
                
            activation = activations[0]
            gradient = gradients[0]
            
            # Grad-CAM logic
            weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
            grad_cam = torch.sum(weights * activation, dim=1).squeeze()
            grad_cam = F.relu(grad_cam).detach().cpu().numpy()
            
            if np.max(grad_cam) > 0:
                grad_cam /= np.max(grad_cam)
                
            # Resize to original image size
            heatmap_resized = cv2.resize(grad_cam, (img.shape[1], img.shape[0]))
            return heatmap_resized
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            return None

    def get_validation_metrics(self, data_yaml):
        """
        Run official validation and return the confusion matrix, mAP, and per-class metrics.
        """
        val_results = self.model.val(data=data_yaml, verbose=False)
        # val_results.confusion_matrix.matrix is the [n_classes+1, n_classes+1] matrix
        cm = val_results.confusion_matrix.matrix
        
        # 1. Overall Metrics
        metrics = {
            'mAP50': val_results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': val_results.results_dict['metrics/mAP50-95(B)'],
            'Precision': val_results.results_dict['metrics/precision(B)'],
            'Recall': val_results.results_dict['metrics/recall(B)'],
            'F1': 2 * (val_results.results_dict['metrics/precision(B)'] * val_results.results_dict['metrics/recall(B)']) / 
                  (val_results.results_dict['metrics/precision(B)'] + val_results.results_dict['metrics/recall(B)'] + 1e-8),
            # Speed metrics in ms per image
            'InferenceTime': val_results.speed['inference'],
            'PreprocessTime': val_results.speed['preprocess'],
            'PostprocessTime': val_results.speed['postprocess']
        }
        
        # 2. Per-Class Metrics
        names = val_results.names # {id: name}
        per_class = {}
        
        # Extract per-class arrays
        ps = np.atleast_1d(val_results.box.p)
        rs = np.atleast_1d(val_results.box.r)
        # all_ap is [n_classes, 10] where index 0 is mAP50
        all_ap = val_results.box.all_ap 

        for i, (class_id, name) in enumerate(sorted(names.items())):
            if i < len(ps) and i < all_ap.shape[0]:
                per_class[name] = {
                    'Precision': ps[i],
                    'Recall': rs[i],
                    'mAP50': all_ap[i, 0],
                    'mAP50-95': all_ap[i].mean()
                }
            
        return metrics, cm, per_class
