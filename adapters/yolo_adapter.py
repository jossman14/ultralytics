import torch
import torch.nn.functional as F
import numpy as np
import cv2
from ultralytics import YOLO
import os

class YOLOv8Adapter:
    """
    Adapter to connect YOLOv8 framework to the Scopus Analytics Core.
    """
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)
        
    def get_predictions(self, img_path):
        """
        Extract confidence scores and bboxes from an image.
        """
        results = self.model.predict(img_path, verbose=False)
        if not results:
            return [], []
            
        res = results[0]
        confs = res.boxes.conf.cpu().numpy()
        bboxes = res.boxes.xyxy.cpu().numpy() # [x1, y1, x2, y2]
        return confs, bboxes

    def generate_eigencam(self, img_path, target_layer_idx=-2):
        """
        Generate EigenCAM heatmap for the given image.
        """
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = F.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.to(self.model.device)
        
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.cpu().detach())
            
        try:
            target_layer = self.model.model.model[target_layer_idx]
            hook = target_layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                self.model.model(img_tensor)
                
            hook.remove()
            
            if not activations:
                return None
                
            activation = activations[0] # [1, C, H, W]
            heatmap = torch.mean(activation, dim=1).squeeze().numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
                
            # Resize to original image size
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            return heatmap_resized
            
        except Exception as e:
            print(f"Error generating EigenCAM: {e}")
            return None

    def get_validation_metrics(self, data_yaml):
        """
        Run official validation and return the confusion matrix and mAP.
        """
        val_results = self.model.val(data=data_yaml, verbose=False)
        # val_results.confusion_matrix.matrix is the [n_classes+1, n_classes+1] matrix
        cm = val_results.confusion_matrix.matrix
        # Standard metrics
        metrics = {
            'mAP50': val_results.results_dict['metrics/mAP50(B)'],
            'mAP50-95': val_results.results_dict['metrics/mAP50-95(B)'],
            'Precision': val_results.results_dict['metrics/precision(B)'],
            'Recall': val_results.results_dict['metrics/recall(B)'],
            'F1': 2 * (val_results.results_dict['metrics/precision(B)'] * val_results.results_dict['metrics/recall(B)']) / 
                  (val_results.results_dict['metrics/precision(B)'] + val_results.results_dict['metrics/recall(B)'] + 1e-8)
        }
        return metrics, cm
