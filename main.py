import cv2
import numpy as np
import math

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.next_object_id = 0
        self.max_objects = 10
        
    def detect_objects(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        
        edges = cv2.Canny(blur, 30, 100)
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 15000:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w > 30 and h > 30 and w < 400 and h < 400:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    contour_points = []
                    for point in approx:
                        px, py = point[0]
                        contour_points.append((px, py))
                    
                    objects.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'contour': contour_points,
                        'area': area
                    })
        
        return objects
    
    def track_objects(self, current_objects):
        if not self.tracked_objects:
            for obj in current_objects:
                if len(self.tracked_objects) < self.max_objects:
                    self.tracked_objects[self.next_object_id] = {
                        'center': obj['center'],
                        'bbox': obj['bbox'],
                        'contour': obj['contour'],
                        'area': obj['area'],
                        'age': 0,
                        'stability': 1,
                        'last_seen': 0
                    }
                    self.next_object_id += 1
            return
        
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['last_seen'] += 1
        
        used_objects = set()
        
        for obj_id in list(self.tracked_objects.keys()):
            best_match = None
            best_distance = float('inf')
            
            for i, current_obj in enumerate(current_objects):
                if i in used_objects:
                    continue
                    
                old_center = self.tracked_objects[obj_id]['center']
                new_center = current_obj['center']
                distance = math.sqrt((old_center[0] - new_center[0])**2 + 
                                   (old_center[1] - new_center[1])**2)
                
                area_diff = abs(self.tracked_objects[obj_id]['area'] - current_obj['area'])
                area_ratio = area_diff / max(self.tracked_objects[obj_id]['area'], current_obj['area'])
                
                if distance < 120 and area_ratio < 0.5 and distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                self.tracked_objects[obj_id].update(current_objects[best_match])
                self.tracked_objects[obj_id]['age'] += 1
                self.tracked_objects[obj_id]['stability'] = min(10, self.tracked_objects[obj_id]['stability'] + 1)
                self.tracked_objects[obj_id]['last_seen'] = 0
                used_objects.add(best_match)
            else:
                self.tracked_objects[obj_id]['stability'] = max(0, self.tracked_objects[obj_id]['stability'] - 2)
        
        objects_to_remove = [obj_id for obj_id, obj_data in self.tracked_objects.items() 
                           if obj_data['stability'] <= 0 or obj_data['last_seen'] > 15]
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        for i, obj in enumerate(current_objects):
            if i not in used_objects and len(self.tracked_objects) < self.max_objects:
                self.tracked_objects[self.next_object_id] = {
                    'center': obj['center'],
                    'bbox': obj['bbox'],
                    'contour': obj['contour'],
                    'area': obj['area'],
                    'age': 0,
                    'stability': 3,
                    'last_seen': 0
                }
                self.next_object_id += 1

class ParticleNetwork:
    def __init__(self):
        self.tracker = ObjectTracker()
        # Futuristic tech blue/cyan color (BGR format)
        self.tech_color = (255, 200, 50)  # A bright cyan-blue color
        
    def create_object_particles(self, contour, num_particles=15):
        if len(contour) < 3:
            return []
            
        particles = []
        
        for i in range(0, len(contour), max(1, len(contour) // num_particles)):
            particles.append(contour[i])
        
        return particles
    
    def draw_object_network(self, frame, obj_id, obj_data):
        if obj_data['stability'] < 3:
            return
            
        contour = obj_data['contour']
        center = obj_data['center']
        
        if len(contour) < 2:
            return
        
        particles = self.create_object_particles(contour)
        
        stability_alpha = min(1.0, obj_data['stability'] / 10.0)
        
        for i, p1 in enumerate(particles):
            x1, y1 = p1
            
            for j, p2 in enumerate(particles):
                if i >= j:
                    continue
                    
                x2, y2 = p2
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if dist < 120:
                    alpha = max(0.3, 1.0 - (dist / 120)) * stability_alpha
                    final_color = tuple(int(c * alpha) for c in self.tech_color)
                    cv2.line(frame, (x1, y1), (x2, y2), final_color, 1)
            
            nearest_to_center = min(particles, 
                                  key=lambda p: math.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2))
            if nearest_to_center != p1:
                x2, y2 = nearest_to_center
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist > 0:
                    alpha = max(0.2, 1.0 - (dist / 150)) * stability_alpha
                    final_color = tuple(int(c * alpha * 0.7) for c in self.tech_color)
                    cv2.line(frame, (x1, y1), (x2, y2), final_color, 1)
        
        particle_color = tuple(int(c * stability_alpha) for c in self.tech_color)
        for x, y in particles:
            cv2.circle(frame, (x, y), 2, particle_color, -1)
        
        cv2.circle(frame, center, 3, (255, 255, 255), -1)
        cv2.circle(frame, center, 2, particle_color, -1)
        
        x, y, w, h = obj_data['bbox']
        cv2.putText(frame, f'ID:{obj_id}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, particle_color, 1)
    
    def process_frame(self, frame):
        objects = self.tracker.detect_objects(frame)
        self.tracker.track_objects(objects)
        
        for obj_id, obj_data in self.tracker.tracked_objects.items():
            self.draw_object_network(frame, obj_id, obj_data)
        
        cv2.putText(frame, f'Objects: {len(self.tracker.tracked_objects)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.tech_color, 2)

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    network = ParticleNetwork()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        network.process_frame(frame)
        
        cv2.imshow('Object Particle Network', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            network.tracker.tracked_objects.clear()
            network.tracker.next_object_id = 0
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()