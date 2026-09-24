import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO


class AccidentDetector:

    def __init__(
        self,
        model_path="yolo11n.pt",
        conf_threshold=0.70,
        iou_threshold=0.20,
        collision_distance_ratio=0.45,
        sudden_stop_ratio=0.35,
        direction_change_threshold=45,
        temporal_buffer_limit=10,
        history_length=12
    ):

        print("Loading YOLO11n...")

        self.model = YOLO(model_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.collision_distance_ratio = collision_distance_ratio
        self.sudden_stop_ratio = sudden_stop_ratio
        self.direction_change_threshold = direction_change_threshold
        self.temporal_buffer_limit = temporal_buffer_limit
        self.history_length = history_length

        # =====================================================
        # COCO classes
        # =====================================================

        self.vehicle_classes = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

        # Other objects which can be relevant around roads
        self.roadside_classes = {
            0: "Person",
            9: "Traffic Light",
            11: "Stop Sign",
            12: "Parking Meter",
            13: "Bench"
        }

        # =====================================================
        # Tracking history
        # =====================================================

        self.track_history = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

        self.speed_history = defaultdict(
            lambda: deque(maxlen=8)
        )

        self.direction_history = defaultdict(
            lambda: deque(maxlen=8)
        )

        # =====================================================
        # Accident state
        # =====================================================

        self.accident_score = 0
        self.consecutive_accident_count = 0

        self.alert_triggered = False

        self.accident_reason = ""
        self.accident_track_ids = []

        self.accident_timestamp = None

        # Prevent repeated triggering
        self.last_alert_time = 0

    # =========================================================
    # IMAGE PREPROCESSING
    # =========================================================

    def enhance_frame(self, frame):

        """
        Mild enhancement for difficult lighting/weather.
        """

        try:

            lab = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2LAB
            )

            l, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )

            l = clahe.apply(l)

            enhanced = cv2.merge(
                (l, a, b)
            )

            enhanced = cv2.cvtColor(
                enhanced,
                cv2.COLOR_LAB2BGR
            )

            return enhanced

        except Exception:

            return frame

    # =========================================================
    # CENTROID
    # =========================================================

    def calculate_centroid(self, box):

        x1, y1, x2, y2 = box

        return (
            (x1 + x2) / 2,
            (y1 + y2) / 2
        )

    # =========================================================
    # BOX SIZE
    # =========================================================

    def box_size(self, box):

        x1, y1, x2, y2 = box

        width = max(
            1,
            x2 - x1
        )

        height = max(
            1,
            y2 - y1
        )

        return np.sqrt(
            width ** 2 +
            height ** 2
        )

    # =========================================================
    # IOU
    # =========================================================

    def calculate_iou(
        self,
        box1,
        box2
    ):

        x1 = max(
            box1[0],
            box2[0]
        )

        y1 = max(
            box1[1],
            box2[1]
        )

        x2 = min(
            box1[2],
            box2[2]
        )

        y2 = min(
            box1[3],
            box2[3]
        )

        intersection_width = max(
            0,
            x2 - x1
        )

        intersection_height = max(
            0,
            y2 - y1
        )

        intersection_area = (
            intersection_width *
            intersection_height
        )

        area1 = (
            max(0, box1[2] - box1[0]) *
            max(0, box1[3] - box1[1])
        )

        area2 = (
            max(0, box2[2] - box2[0]) *
            max(0, box2[3] - box2[1])
        )

        union = (
            area1 +
            area2 -
            intersection_area
        )

        if union <= 0:
            return 0.0

        return intersection_area / union

    # =========================================================
    # DISTANCE BETWEEN VEHICLES
    # =========================================================

    def calculate_distance_ratio(
        self,
        box1,
        box2
    ):

        center1 = self.calculate_centroid(
            box1
        )

        center2 = self.calculate_centroid(
            box2
        )

        distance = np.linalg.norm(
            np.array(center1) -
            np.array(center2)
        )

        size1 = self.box_size(box1)
        size2 = self.box_size(box2)

        average_size = (
            size1 +
            size2
        ) / 2

        if average_size <= 0:
            return 999

        return distance / average_size

    # =========================================================
    # ANGLE BETWEEN VECTORS
    # =========================================================

    def angle_between_vectors(
        self,
        v1,
        v2
    ):

        v1 = np.array(v1)
        v2 = np.array(v2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0

        cosine = np.dot(
            v1,
            v2
        ) / (
            norm1 *
            norm2
        )

        cosine = np.clip(
            cosine,
            -1,
            1
        )

        return np.degrees(
            np.arccos(cosine)
        )

    # =========================================================
    # UPDATE TRACK HISTORY
    # =========================================================

    def update_track(
        self,
        track_id,
        center
    ):

        history = self.track_history[
            track_id
        ]

        history.append(
            center
        )

        if len(history) >= 2:

            previous = np.array(
                history[-2]
            )

            current = np.array(
                history[-1]
            )

            movement = np.linalg.norm(
                current -
                previous
            )

            self.speed_history[
                track_id
            ].append(
                movement
            )

            direction = (
                current -
                previous
            )

            self.direction_history[
                track_id
            ].append(
                direction
            )

    # =========================================================
    # SUDDEN STOP DETECTION
    # =========================================================

    def detect_sudden_stop(
        self,
        track_id
    ):

        speeds = self.speed_history[
            track_id
        ]

        if len(speeds) < 5:
            return False

        previous_speeds = list(
            speeds
        )[:-1]

        current_speed = speeds[-1]

        average_previous = np.mean(
            previous_speeds
        )

        if average_previous < 3:
            return False

        reduction = (
            1 -
            current_speed /
            average_previous
        )

        return (
            reduction >=
            self.sudden_stop_ratio
        )

    # =========================================================
    # SUDDEN DIRECTION CHANGE
    # =========================================================

    def detect_direction_change(
        self,
        track_id
    ):

        directions = self.direction_history[
            track_id
        ]

        if len(directions) < 4:
            return False

        previous = directions[-2]
        current = directions[-1]

        angle = self.angle_between_vectors(
            previous,
            current
        )

        return (
            angle >=
            self.direction_change_threshold
        )

    # =========================================================
    # ABNORMAL MOVEMENT
    # =========================================================

    def detect_abnormal_motion(
        self,
        track_id
    ):

        history = self.track_history[
            track_id
        ]

        if len(history) < 6:
            return False

        points = list(history)

        movements = []

        for i in range(
            1,
            len(points)
        ):

            p1 = np.array(
                points[i - 1]
            )

            p2 = np.array(
                points[i]
            )

            movements.append(
                np.linalg.norm(
                    p2 - p1
                )
            )

        if len(movements) < 5:
            return False

        recent = movements[-2:]
        old = movements[:-2]

        old_average = np.mean(old)

        if old_average < 2:
            return False

        recent_average = np.mean(
            recent
        )

        if recent_average > (
            old_average * 3.0
        ):
            return True

        return False

    # =========================================================
    # VEHICLE COLLISION
    # =========================================================

    def detect_vehicle_collision(
        self,
        vehicles
    ):

        collisions = []

        for i in range(
            len(vehicles)
        ):

            for j in range(
                i + 1,
                len(vehicles)
            ):

                vehicle1 = vehicles[i]
                vehicle2 = vehicles[j]

                box1 = vehicle1["box"]
                box2 = vehicle2["box"]

                iou = self.calculate_iou(
                    box1,
                    box2
                )

                distance_ratio = (
                    self.calculate_distance_ratio(
                        box1,
                        box2
                    )
                )

                if iou >= self.iou_threshold:

                    collisions.append(
                        {
                            "type":
                                "Vehicle collision",

                            "ids":
                                [
                                    vehicle1["id"],
                                    vehicle2["id"]
                                ],

                            "iou":
                                iou,

                            "distance":
                                distance_ratio
                        }
                    )

                    continue

                if (
                    distance_ratio <=
                    self.collision_distance_ratio
                ):

                    collisions.append(
                        {
                            "type":
                                "Possible vehicle collision",

                            "ids":
                                [
                                    vehicle1["id"],
                                    vehicle2["id"]
                                ],

                            "iou":
                                iou,

                            "distance":
                                distance_ratio
                        }
                    )

        return collisions

    # =========================================================
    # ROAD-SIDE OBJECT INTERACTION
    # =========================================================

    def detect_object_interaction(
        self,
        vehicles,
        objects
    ):

        events = []

        for vehicle in vehicles:

            vehicle_box = vehicle["box"]

            for obj in objects:

                object_box = obj["box"]

                iou = self.calculate_iou(
                    vehicle_box,
                    object_box
                )

                distance = (
                    self.calculate_distance_ratio(
                        vehicle_box,
                        object_box
                    )
                )

                if (
                    iou >= 0.05 or
                    distance <= 0.55
                ):

                    events.append(
                        {
                            "type":
                                "Vehicle-roadside object interaction",

                            "vehicle_id":
                                vehicle["id"],

                            "object":
                                obj["name"],

                            "iou":
                                iou,

                            "distance":
                                distance
                        }
                    )

        return events

    # =========================================================
    # DRAW BOX
    # =========================================================

    def draw_box(
        self,
        frame,
        box,
        color,
        label
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

    # =========================================================
    # MAIN FRAME PROCESSOR
    # =========================================================

    def process_frame(
        self,
        frame,
        timestamp=0
    ):

        original_frame = frame.copy()

        # =====================================================
        # Mild enhancement
        # =====================================================

        enhanced_frame = self.enhance_frame(
            frame
        )

        # =====================================================
        # YOLO + ByteTrack
        # =====================================================

        results = self.model.track(
            enhanced_frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            verbose=False
        )

        vehicles = []
        objects = []

        if (
            results and
            results[0].boxes is not None
        ):

            boxes = results[0].boxes

            xyxy = boxes.xyxy.cpu().numpy()

            classes = boxes.cls.cpu().numpy()

            confidences = boxes.conf.cpu().numpy()

            if boxes.id is not None:

                track_ids = (
                    boxes.id.cpu()
                    .numpy()
                    .astype(int)
                )

            else:

                track_ids = [
                    -1
                ] * len(xyxy)

            # =================================================
            # PROCESS DETECTIONS
            # =================================================

            for box, cls, conf, track_id in zip(
                xyxy,
                classes,
                confidences,
                track_ids
            ):

                class_id = int(cls)

                confidence = float(
                    conf
                )

                # ---------------------------------------------
                # Vehicle
                # ---------------------------------------------

                if class_id in self.vehicle_classes:

                    center = (
                        self.calculate_centroid(
                            box
                        )
                    )

                    if track_id >= 0:

                        self.update_track(
                            track_id,
                            center
                        )

                    vehicles.append(
                        {
                            "box":
                                box,

                            "id":
                                int(track_id),

                            "class":
                                self.vehicle_classes[
                                    class_id
                                ],

                            "confidence":
                                confidence
                        }
                    )

                # ---------------------------------------------
                # Roadside objects
                # ---------------------------------------------

                elif class_id in self.roadside_classes:

                    objects.append(
                        {
                            "box":
                                box,

                            "id":
                                int(track_id),

                            "name":
                                self.roadside_classes[
                                    class_id
                                ],

                            "confidence":
                                confidence
                        }
                    )

        # =====================================================
        # EVENT DETECTION
        # =====================================================

        detected_events = []

        accident_ids = []

        # -----------------------------------------------------
        # 1. Vehicle collision
        # -----------------------------------------------------

        collision_events = (
            self.detect_vehicle_collision(
                vehicles
            )
        )

        if collision_events:

            for event in collision_events:

                detected_events.append(
                    event["type"]
                )

                accident_ids.extend(
                    event["ids"]
                )

        # -----------------------------------------------------
        # 2. Roadside interaction
        # -----------------------------------------------------

        object_events = (
            self.detect_object_interaction(
                vehicles,
                objects
            )
        )

        if object_events:

            for event in object_events:

                detected_events.append(
                    event["type"]
                )

                accident_ids.append(
                    event["vehicle_id"]
                )

        # -----------------------------------------------------
        # 3. Sudden stop
        # -----------------------------------------------------

        for vehicle in vehicles:

            track_id = vehicle["id"]

            if track_id < 0:
                continue

            if self.detect_sudden_stop(
                track_id
            ):

                detected_events.append(
                    "Sudden vehicle stop"
                )

                accident_ids.append(
                    track_id
                )

        # -----------------------------------------------------
        # 4. Sudden direction change
        # -----------------------------------------------------

        for vehicle in vehicles:

            track_id = vehicle["id"]

            if track_id < 0:
                continue

            if self.detect_direction_change(
                track_id
            ):

                detected_events.append(
                    "Sudden direction change"
                )

                accident_ids.append(
                    track_id
                )

        # -----------------------------------------------------
        # 5. Abnormal motion
        # -----------------------------------------------------

        for vehicle in vehicles:

            track_id = vehicle["id"]

            if track_id < 0:
                continue

            if self.detect_abnormal_motion(
                track_id
            ):

                detected_events.append(
                    "Abnormal vehicle movement"
                )

                accident_ids.append(
                    track_id
                )

        # =====================================================
        # REMOVE DUPLICATES
        # =====================================================

        detected_events = list(
            dict.fromkeys(
                detected_events
            )
        )

        accident_ids = list(
            dict.fromkeys(
                accident_ids
            )
        )

        # =====================================================
        # TEMPORAL ACCIDENT REASONING
        # =====================================================

        if detected_events:

            self.accident_score += 1

            self.consecutive_accident_count += 1

        else:

            self.accident_score = max(
                0,
                self.accident_score - 1
            )

            self.consecutive_accident_count = max(
                0,
                self.consecutive_accident_count - 1
            )

        # =====================================================
        # CONFIRM ACCIDENT
        # =====================================================

        confirmed_now = False

        if (
            self.consecutive_accident_count >=
            self.temporal_buffer_limit
        ):

            if not self.alert_triggered:

                self.alert_triggered = True

                self.accident_timestamp = timestamp

                self.accident_reason = (
                    " + ".join(
                        detected_events
                    )
                )

                self.accident_track_ids = (
                    accident_ids
                )

                confirmed_now = True

        # =====================================================
        # DRAW NORMAL VEHICLES
        # =====================================================

        output = original_frame.copy()

        for vehicle in vehicles:

            track_id = vehicle["id"]

            label = (
                f'{vehicle["class"]} '
                f'ID:{track_id}'
            )

            self.draw_box(
                output,
                vehicle["box"],
                (255, 255, 0),
                label
            )

        # =====================================================
        # DRAW RELEVANT OBJECTS
        # =====================================================

        for obj in objects:

            self.draw_box(
                output,
                obj["box"],
                (255, 200, 0),
                obj["name"]
            )

        # =====================================================
        # HIGHLIGHT POSSIBLE ACCIDENT VEHICLES
        # =====================================================

        for vehicle in vehicles:

            if vehicle["id"] in accident_ids:

                self.draw_box(
                    output,
                    vehicle["box"],
                    (0, 0, 255),
                    "POSSIBLE ACCIDENT"
                )

        # =====================================================
        # STATUS
        # =====================================================

        if self.alert_triggered:

            cv2.putText(
                output,
                "ACCIDENT DETECTED!",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3
            )

            cv2.putText(
                output,
                self.accident_reason[:80],
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        elif detected_events:

            cv2.putText(
                output,
                "POSSIBLE ACCIDENT / ABNORMAL EVENT",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 165, 255),
                2
            )

            cv2.putText(
                output,
                " | ".join(
                    detected_events
                )[:90],
                (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2
            )

        else:

            cv2.putText(
                output,
                "MONITORING TRAFFIC",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # =====================================================
        # RETURN INFORMATION
        # =====================================================

        information = {

            "vehicles":
                len(vehicles),

            "events":
                detected_events,

            "accident_ids":
                accident_ids,

            "buffer":
                self.consecutive_accident_count,

            "confirmed":
                self.alert_triggered,

            "confirmed_now":
                confirmed_now,

            "reason":
                self.accident_reason,

            "timestamp":
                self.accident_timestamp
        }

        return (
            output,
            self.alert_triggered,
            self.consecutive_accident_count,
            information
        )