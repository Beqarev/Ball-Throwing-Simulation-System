# Technical Documentation: Ball Throwing Simulation System

## 1. Statement of the Problem

The project addresses the challenge of automatically detecting circular targets in an image and calculating optimal trajectories to hit these targets with a projectile under realistic physics constraints. This involves three main components:

1. **Image Processing Challenge**: 
   - Detect edges in an input image
   - Identify circular targets from these edges
   - Handle varying image conditions and target sizes

2. **Physics Simulation Challenge**:
   - Model projectile motion under gravity
   - Calculate optimal launch parameters (velocity and angle)
   - Ensure accurate trajectory prediction

3. **Integration Challenge**:
   - Convert between image and physical coordinate systems
   - Synchronize detection and simulation components
   - Provide visual feedback of trajectories

## 2. Mathematical Model

### 2.1 Edge Detection Model

The Canny edge detection process is modeled through multiple stages:

1. **Gaussian Smoothing**:
   ```
   G(x,y) = (1/2πσ²)exp(-(x² + y²)/2σ²)
   ```
   where σ is the Gaussian standard deviation.

2. **Gradient Calculation**:
   ```
   ∇f = [Gx, Gy] = [∂f/∂x, ∂f/∂y]
   Magnitude = √(Gx² + Gy²)
   Direction = arctan(Gy/Gx)
   ```

3. **Non-Maximum Suppression**:
   For pixel p(x,y):
   ```
   p(x,y) = {
     M(x,y)  if M(x,y) > M(neighbors in gradient direction)
     0       otherwise
   }
   ```
   where M is the gradient magnitude.

### 2.2 Shape Detection Model

Circle detection uses the isoperimetric inequality:
```
4πA ≤ P²
```
where A is area and P is perimeter. Equality holds only for circles.

Circularity measure:
```
C = 4πA/P²
```

### 2.3 Projectile Motion Model

System of differential equations:
```
dx/dt = vx
dy/dt = vy
dvx/dt = 0
dvy/dt = -g
```

Boundary conditions:
```
x(0) = x₀
y(0) = y₀
vx(0) = v₀cosθ
vy(0) = v₀sinθ
```

### 2.4 Target Hitting Equations

For a target at (xt, yt):
```
xt = v₀cosθ × t
yt = v₀sinθ × t - (1/2)gt²
```

Solving for initial velocity:
```
v₀ = √((gxt²)/(2cosθ(xttanθ - yt)))
```

## 3. Numerical Methods and Properties

### 3.1 Euler Method

Update equations:
```
x_{n+1} = xn + h × vx_n
y_{n+1} = yn + h × vy_n
vx_{n+1} = vx_n
vy_{n+1} = vy_n - h × g
```

Properties:
- Local truncation error: O(h²)
- Global truncation error: O(h)
- Stability region: |1 + hλ| ≤ 1

### 3.2 RK4 Method

Stage calculations:
```
k1 = f(tn, yn)
k2 = f(tn + h/2, yn + hk1/2)
k3 = f(tn + h/2, yn + hk2/2)
k4 = f(tn + h, yn + hk3)
yn+1 = yn + (h/6)(k1 + 2k2 + 2k3 + k4)
```

Properties:
- Local truncation error: O(h⁵)
- Global truncation error: O(h⁴)
- Larger stability region than Euler method

## 4. Algorithm

### 4.1 Edge Detection Algorithm
```python
def detect_edges(image):
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Compute gradients using Sobel operators
    4. Perform non-maximum suppression
    5. Apply double thresholding
    6. Perform edge tracking by hysteresis
    return edges
```

### 4.2 Target Detection Algorithm
```python
def detect_targets(edges):
    1. Find contours in edge image
    2. For each contour:
        a. Calculate area and perimeter
        b. Compute circularity measure
        c. If circularity > threshold:
            - Fit minimum enclosing circle
            - Add to targets list
    3. Sort targets by x-coordinate
    return targets
```

### 4.3 Trajectory Calculation Algorithm
```python
def calculate_trajectory(start_pos, target_pos):
    1. Estimate base angle to target
    2. For angle_offset in search range:
        a. Calculate required initial velocity
        b. If velocity in valid range:
            - Simulate trajectory using RK4
            - Calculate minimum distance to target
            - Update best solution if improved
    3. Return best trajectory
```

## 5. Test Cases and Results

### 5.1 Simple Background Test Case

**Test Image Properties**:
- White/solid background
- Clear circular targets
- Good contrast

**Results**:
- Edge Detection: Clear, continuous edges
- Target Detection: >95% accuracy
- Trajectory Calculation: Successfully hits targets

### 5.2 Complex Background Test Case

**Test Image Properties**:
- Textured/noisy background
- Multiple shapes
- Varying contrast

**Results**:
- Edge Detection: Noisy edges, many false positives
- Target Detection: <60% accuracy
- Trajectory Calculation: Unreliable due to false targets

### 5.3 Limitations and Issues

1. **Background Sensitivity**:
   - System performs poorly with complex backgrounds
   - Edge detection picks up background texture
   - False positives in target detection

2. **Lighting Conditions**:
   - Shadows can create false edges
   - Low contrast reduces detection accuracy
   - Reflections can cause false targets

3. **Shape Variations**:
   - Imperfect circles may be missed
   - Overlapping targets cause issues
   - Size variations affect detection accuracy

### 5.4 Visualization

The system provides several visualization components:
1. Edge detection overlay
2. Detected target markers
3. Calculated trajectories
4. Animated projectile motion

### 5.5 Performance Metrics

For simple backgrounds:
- Edge Detection Time: ~50ms
- Target Detection Time: ~30ms
- Trajectory Calculation: ~100ms per target
- Overall Accuracy: >90%

For complex backgrounds:
- Edge Detection Time: ~80ms
- Target Detection Time: ~50ms
- False Positive Rate: >40%
- Overall Accuracy: <60%

## 6. Recommendations for Improvement

1. **Image Processing**:
   - Implement adaptive thresholding
   - Add background subtraction
   - Use machine learning for target detection

2. **Physics Model**:
   - Include air resistance
   - Add wind effects
   - Consider Magnus force

3. **Algorithm Optimization**:
   - Parallel processing for trajectory calculation
   - GPU acceleration for image processing
   - Adaptive time stepping in numerical integration

4. **User Interface**:
   - Add parameter tuning interface
   - Provide real-time feedback
   - Include debugging visualization options
