import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from EdgeDetection import CannyEdgeDetector

class ShapeDetector:
    def __init__(self, min_area: int = 100, epsilon_factor: float = 0.01, circle_threshold: float = 0.8):
        self.min_area = min_area
        self.epsilon_factor = epsilon_factor
        self.circle_threshold = circle_threshold

    def detect_shapes(self, edges: np.ndarray) -> List[Tuple[int, int, int]]:
        shapes = []
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue

            epsilon = self.epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if self._is_circle(contour):
                shapes.append((center[0], center[1], radius))
            else:
                shapes.append((center[0], center[1], radius))

        return shapes

    def _is_circle(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            return False

        circularity = 4 * np.pi * area / (perimeter ** 2)

        return circularity > self.circle_threshold

    def draw_circles(self, image: np.ndarray, shapes: List[Tuple[int, int, int]]) -> np.ndarray:
        output_image = image.copy()
        for (x, y, radius) in shapes:
            cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
        return output_image

if __name__ == "__main__":
    image = cv2.imread("Untitled design.png", cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Image not found")

    detector = CannyEdgeDetector(low_threshold=0.05, high_threshold=0.2)

    edges = detector.detect(image)

    shape_detector = ShapeDetector(min_area=100, epsilon_factor=0.01, circle_threshold=0.8)

    shapes = shape_detector.detect_shapes(edges)

    output_image = shape_detector.draw_circles(image, shapes)

    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Shapes with Circles")
    plt.show()