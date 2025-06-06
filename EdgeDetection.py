import cv2
import numpy as np
import matplotlib.pyplot as plt

class CannyEdgeDetector:
    def __init__(self, low_threshold: float = 0.08, high_threshold: float = 0.25,
                 gaussian_kernel_size: int = 5, gaussian_sigma: float = 1.4):
        if not 0 <= low_threshold <= high_threshold <= 1:
            raise ValueError("Thresholds must be in range [0,1] and low_threshold <= high_threshold")
        if gaussian_kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel size must be odd")
        if gaussian_sigma <= 0:
            raise ValueError("Gaussian sigma must be positive")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    def create_gaussian_kernel(self) -> np.ndarray:
        size = self.gaussian_kernel_size
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        gaussian = np.exp(-((x ** 2 + y ** 2) / (2.0 * self.gaussian_sigma ** 2)))
        return gaussian / gaussian.sum()

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kernel_height, kernel_width = kernel.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

        output = np.zeros_like(image)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                output[i, j] = np.sum(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

        return output

    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        kernel = self.create_gaussian_kernel()
        return self.convolve2d(image, kernel)

    def compute_gradients(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

        gradient_x = self.convolve2d(image, sobel_x)
        gradient_y = self.convolve2d(image, sobel_y)

        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction

    @staticmethod
    def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        height, width = magnitude.shape
        angle = np.degrees(direction) % 180
        suppressed = np.zeros_like(magnitude)

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    suppressed[i, j] = magnitude[i, j]
                else:
                    suppressed[i, j] = 0

        return suppressed

    def double_threshold(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        high = image.max() * self.high_threshold
        low = high * self.low_threshold
        strong_edges = (image >= high)
        weak_edges = (image >= low) & (image < high)
        return strong_edges, weak_edges

    @staticmethod
    def edge_tracking(strong_edges: np.ndarray, weak_edges: np.ndarray) -> np.ndarray:
        result = strong_edges.copy()
        height, width = strong_edges.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if weak_edges[i, j]:
                    if (result[i - 1, j - 1] or result[i - 1, j] or result[i - 1, j + 1] or
                        result[i, j - 1] or result[i, j + 1] or
                        result[i + 1, j - 1] or result[i + 1, j] or result[i + 1, j + 1]):
                        result[i, j] = True

        return result

    def detect(self, image: np.ndarray) -> np.ndarray:
        gray = self.to_grayscale(image)

        blurred = self.apply_gaussian_blur(gray)

        gradient_magnitude, gradient_direction = self.compute_gradients(blurred)

        suppressed = self.non_maximum_suppression(gradient_magnitude, gradient_direction)

        strong_edges, weak_edges = self.double_threshold(suppressed)

        edges = self.edge_tracking(strong_edges, weak_edges)

        return edges.astype(np.uint8) * 255


if __name__ == "__main__":
    image = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Image not found")

    detector = CannyEdgeDetector(low_threshold=0.08, high_threshold=0.25)

    edges = detector.detect(image)

    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection (Custom Convolution)")
    plt.show()